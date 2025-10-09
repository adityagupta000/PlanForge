"""
Dynamic Multi-stage training system with adaptive curriculum learning
Implements novel training strategies: dynamic stage transitions, topology-aware scheduling,
multi-objective optimization, and cross-modal consistency learning
"""

import torch
import torch.nn.utils

# training/trainer.py - Fixed AMP imports
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from collections import deque

from .losses import ResearchGradeLoss, LossScheduler
from config import DEFAULT_TRAINING_CONFIG, DEFAULT_LOSS_CONFIG, StageTransitionCriteria


class CurriculumState:
    """Tracks curriculum learning state and metrics"""

    def __init__(self, config):
        self.config = config

        # Loss history for plateau detection
        self.loss_history = {
            "stage1": deque(maxlen=config.plateau_detection_window * 2),
            "stage2": deque(maxlen=config.plateau_detection_window * 2),
            "stage3": deque(maxlen=config.plateau_detection_window * 2),
        }

        # Component loss tracking
        self.component_losses = {
            "segmentation": deque(maxlen=20),
            "dice": deque(maxlen=20),
            "polygon": deque(maxlen=20),
            "voxel": deque(maxlen=20),
            "topology": deque(maxlen=20),
            "latent_consistency": deque(maxlen=20),
            "graph": deque(maxlen=20),
        }

        # Gradient magnitude tracking for dynamic weighting
        self.gradient_norms = {
            name: deque(maxlen=config.gradient_norm_window)
            for name in self.component_losses.keys()
        }

        # Stage transition tracking
        self.epochs_without_improvement = 0
        self.best_val_loss = float("inf")
        self.stage_transition_epochs = []

        # Dynamic weights history
        self.weight_history = []

    def update_loss_history(self, stage: str, val_loss: float):
        """Update validation loss history for plateau detection"""
        if stage in self.loss_history:
            self.loss_history[stage].append(val_loss)

        # Update improvement tracking
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def update_component_losses(self, loss_components: Dict[str, float]):
        """Update individual loss component history"""
        for name, loss_val in loss_components.items():
            if name in self.component_losses:
                self.component_losses[name].append(loss_val)

    def should_transition(self, current_stage: int) -> bool:
        """Check if should transition to next stage"""
        if current_stage == 1:
            val_losses = list(self.loss_history["stage1"])
            return StageTransitionCriteria.should_transition_from_stage1(
                [], val_losses, self.config
            )
        elif current_stage == 2:
            polygon_losses = list(self.component_losses["polygon"])
            return StageTransitionCriteria.should_transition_from_stage2(
                polygon_losses, self.config
            )

        return False


class AdaptiveMultiStageTrainer:
    """
    Advanced multi-stage trainer with dynamic curriculum learning:
    - Adaptive stage transitioning based on performance plateaus
    - Topology-aware loss scheduling
    - Multi-objective optimization with dynamic weighting
    - Cross-modal latent consistency learning
    - Graph-based topology constraints
    """

    # Class constant for rolling checkpoint path
    ROLLING_CHECKPOINT = "latest_checkpoint.pth"

    def __init__(self, model, train_loader, val_loader, device=None, config=None):
        if config is None:
            config = DEFAULT_TRAINING_CONFIG

        self.model = model.to(device or config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.device
        self.config = config

        # Initialize curriculum state
        self.curriculum_state = CurriculumState(config.curriculum)
        self.loss_scheduler = LossScheduler(config.curriculum)

        # Training state tracking for resume functionality
        self.current_stage = 1
        self.current_epoch = 0
        self.global_epoch = 0
        self.stage_epoch = 0
        self.stage_start_time = None
        self.epoch_times = []

        # Add AMP and optimization settings - Updated for new PyTorch API
        self.use_amp = getattr(config, "use_mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.accumulation_steps = getattr(config, "accumulation_steps", 1)
        self.dvx_step_freq = getattr(config, "dvx_step_freq", 1)
        self.voxel_size_stage = getattr(config, "voxel_size_stage", None)
        self.image_size_stage = getattr(config, "image_size_stage", None)
        self._step = 0

        # Enhanced optimizers with better hyperparameters
        self.optimizer_2d = torch.optim.AdamW(
            list(self.model.encoder.parameters())
            + list(self.model.seg_head.parameters())
            + list(self.model.attr_head.parameters())
            + list(self.model.sdf_head.parameters()),
            lr=config.stage1_lr,
            weight_decay=config.stage1_weight_decay,
            betas=(0.9, 0.999),
        )

        self.optimizer_dvx = torch.optim.AdamW(
            self.model.dvx.parameters(),
            lr=config.stage2_lr,
            weight_decay=config.stage2_weight_decay,
            betas=(0.9, 0.999),
        )

        self.optimizer_full = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.stage3_lr,
            weight_decay=config.stage3_weight_decay,
            betas=(0.9, 0.999),
        )

        # Enhanced learning rate schedulers with proper minimum LR
        if config.use_cosine_restarts:
            self.scheduler_2d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_2d, T_0=20, T_mult=1,
                eta_min=config.stage1_lr * 0.1  # Min LR is 10% of initial
            )
            self.scheduler_dvx = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_dvx, T_0=15, T_mult=1,
                eta_min=config.stage2_lr * 0.1
            )
            self.scheduler_full = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_full, T_0=30, T_mult=1,
                eta_min=config.stage3_lr * 0.1
            )
        else:
            self.scheduler_2d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_2d, T_max=config.max_stage1_epochs,
                eta_min=config.stage1_lr * 0.1  # Min LR is 10% of initial
            )
            self.scheduler_dvx = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_dvx, T_max=config.max_stage2_epochs,
                eta_min=config.stage2_lr * 0.1
            )
            self.scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_full, T_max=config.max_stage3_epochs,
                eta_min=config.stage3_lr * 0.1
            )

        # Enhanced loss function with dynamic weighting
        base_loss_kwargs = {
            k: v
            for k, v in DEFAULT_LOSS_CONFIG.__dict__.items()
            if k != "enable_dynamic_weighting"
        }
        self.loss_fn = ResearchGradeLoss(
            **base_loss_kwargs,
            enable_dynamic_weighting=bool(config.curriculum.use_gradnorm),
            gradnorm_alpha=float(config.curriculum.gradnorm_alpha),
            device=self.device,
        )

        self.history = {
            "stage1": {"train_loss": [], "val_loss": [], "component_losses": []},
            "stage2": {"train_loss": [], "val_loss": [], "component_losses": []},
            "stage3": {"train_loss": [], "val_loss": [], "component_losses": []},
            "stage_transitions": [],
            "dynamic_weights": [],
            "curriculum_events": [],
        }

    def _get_eta_string(self, epoch, total_epochs):
        """Calculate and format ETA string"""
        if len(self.epoch_times) == 0:
            return "ETA: calculating..."

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = total_epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs

        if eta_seconds < 60:
            return f"ETA: {int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            return f"ETA: {hours}h {minutes}m"

    def _get_shared_parameters(self):
        """Get shared parameters for GradNorm weighting"""
        # Return encoder parameters as shared across tasks
        return list(self.model.encoder.parameters())

    def _update_loss_weights_for_curriculum(
        self, current_stage: int, stage_epoch: int, total_stage_epochs: int
    ):
        """Update loss weights based on curriculum schedule"""
        base_weights = {
            "seg": self.loss_fn.initial_weights["seg"],
            "dice": self.loss_fn.initial_weights["dice"],
            "sdf": self.loss_fn.initial_weights["sdf"],
            "attr": self.loss_fn.initial_weights["attr"],
            "polygon": self.loss_fn.initial_weights["polygon"],
            "voxel": self.loss_fn.initial_weights["voxel"],
            "topology": self.loss_fn.initial_weights["topology"],
            "latent_consistency": self.loss_fn.initial_weights["latent_consistency"],
            "graph": self.loss_fn.initial_weights["graph"],
        }

        scheduled_weights = self.loss_scheduler.get_scheduled_weights(
            current_stage,
            self.global_epoch,
            stage_epoch,
            total_stage_epochs,
            base_weights,
        )

        self.loss_fn.update_loss_weights(scheduled_weights)

        # Log weight changes
        self.history["dynamic_weights"].append(
            {
                "epoch": self.global_epoch,
                "stage": current_stage,
                "weights": scheduled_weights.copy(),
            }
        )

    def _train_epoch(self, mode="stage1"):
        """Enhanced training epoch with improved stability and speed"""
        self.model.train()
        total_loss = 0
        component_loss_sums = {}

        # Select optimizer and apply gradient scaling
        if mode == "stage1":
            optimizer = self.optimizer_2d
        elif mode == "stage2":
            optimizer = self.optimizer_dvx
        else:
            optimizer = self.optimizer_full

        # Improved progress tracking
        train_pbar = tqdm(
            self.train_loader, desc=f"Training {mode.upper()}", leave=False, ncols=120
        )

        batch_count = 0
        epoch_start_time = time.time()
        
        # Add gradient accumulation tracking
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(train_pbar):
            self._step += 1
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Smart geometric computation gating
            run_full_geometric = (
                mode == "stage3" or  # Always run in final stage
                (mode == "stage2" and self._step % 1 == 0) or  # Every other step in stage 2
                (mode == "stage1" and self._step % 2 == 0)  # Every 4th step in stage 1
            )

            with autocast(enabled=self.use_amp):

                predictions = self.model(
                    batch["image"], run_full_geometric=run_full_geometric
                )

                targets = self._prepare_targets(batch, mode)
                
                shared_params = (
                    self._get_shared_parameters()
                    if self.config.curriculum.use_gradnorm
                    else None
                )

                loss, loss_components = self.loss_fn(
                    predictions,
                    targets,
                    shared_params,
                    run_full_geometric=run_full_geometric,
                )

                # Scale for accumulation
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps
                accumulated_loss += loss.item()

            # Backward pass with stability
            self.scaler.scale(loss).backward()

            # Gradient accumulation and update
            # Gradient accumulation and update
            if ((batch_idx + 1) % self.accumulation_steps) == 0:
                # Check if there are any gradients to unscale
                has_grads = any(
                    p.grad is not None 
                    for p in self.model.parameters() 
                    if p.requires_grad
                )
    
                if has_grads:
                    # Enhanced gradient clipping
                    self.scaler.unscale_(optimizer)
                    
                    # Adaptive gradient clipping based on loss magnitude
                    max_grad_norm = min(self.config.grad_clip_norm * (1.0 + accumulated_loss), 2.0)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Skip optimizer step if no gradients
                    pass
                
                optimizer.zero_grad()
                
                # Reset accumulation
                accumulated_loss = 0.0

            current_loss = loss.item() * (self.accumulation_steps if self.accumulation_steps > 1 else 1)
            total_loss += current_loss

            # Track components with better averaging
            for name, component_loss in loss_components.items():
                if name != "total":
                    loss_val = (
                        component_loss.item()
                        if torch.is_tensor(component_loss)
                        else component_loss
                    )
                    if name not in component_loss_sums:
                        component_loss_sums[name] = []
                    component_loss_sums[name].append(loss_val)

            batch_count += 1

            # Less frequent but more informative logging
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - epoch_start_time
                avg_time_per_batch = elapsed / (batch_idx + 1)
                
                # Show meaningful component averages
                recent_components = {}
                for name, vals in component_loss_sums.items():
                    if len(vals) >= 10:  # Only show if we have enough samples
                        recent_avg = np.mean(vals[-10:])  # Last 10 batches
                        if recent_avg > 0.01:  # Only show significant components
                            recent_components[name] = recent_avg
                
                comp_str = ", ".join([f"{k}:{v:.3f}" for k, v in recent_components.items()])
                print(f"[Epoch {self.global_epoch}] Batch {batch_idx+1} | "
                      f"{avg_time_per_batch:.2f}s/batch | loss:{total_loss/batch_count:.4f} | {comp_str}")

            # Update progress with meaningful info
            train_pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # Calculate proper component averages
        avg_component_losses = {}
        for name, loss_list in component_loss_sums.items():
            if loss_list:
                avg_component_losses[name] = np.mean(loss_list)
            else:
                avg_component_losses[name] = 0.0

        return total_loss / batch_count, avg_component_losses
  
    def _prepare_targets(self, batch, mode):
        """Prepare targets based on training mode"""
        if mode == "stage1":
            return {"mask": batch["mask"], "attributes": batch["attributes"]}
        elif mode == "stage2":
            if "polygons_gt" not in batch:
                print(f"Warning: polygons_gt missing in batch for {mode}")
                return {"mask": batch.get("mask"), "attributes": batch.get("attributes")}
        
            return {
                "polygons_gt": {
                    "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                    "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                }
            }
        else:  # stage3
            targets = {
                "mask": batch["mask"],
                "attributes": batch["attributes"],
            }
        
            if "voxels_gt" in batch:
                targets["voxels_gt"] = batch["voxels_gt"]
        
            if "polygons_gt" in batch:
             targets["polygons_gt"] = {
                "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                }
        
            return targets

    def _validate(self, mode="stage1"):
        """Enhanced validation with consistent loss computation"""
        self.model.eval()
        total_loss = 0
        component_loss_sums = {}

        val_pbar = tqdm(
            self.val_loader, desc=f"Validating {mode.upper()}", leave=False, ncols=120
        )

        batch_count = 0
        with torch.no_grad():
            for batch in val_pbar:
                batch = {
                    k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                with autocast(enabled=self.use_amp):

                    # ALWAYS run full geometric in validation for consistency
                    run_full = (mode == "stage3")
                    predictions = self.model(batch["image"], run_full_geometric=True)

                    # Use appropriate targets for current stage
                    targets = self._prepare_targets(batch, mode)  # Use full targets

                    # Use same loss computation as training but without dynamic weighting
                    loss, loss_components = self.loss_fn(
                        predictions, targets, shared_parameters=None, run_full_geometric=True
                    )

                current_loss = loss.item()
                total_loss += current_loss

                # Track component losses properly
                for name, component_loss in loss_components.items():
                    if name != "total":
                        loss_val = (
                            component_loss.item()
                            if torch.is_tensor(component_loss)
                            else component_loss
                        )
                        if name not in component_loss_sums:
                            component_loss_sums[name] = []
                        component_loss_sums[name].append(loss_val)

                batch_count += 1
                val_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Calculate proper averages
        avg_component_losses = {}
        for name, loss_list in component_loss_sums.items():
            if loss_list:
                avg_component_losses[name] = np.mean(loss_list)
            else:
                avg_component_losses[name] = 0.0

        return total_loss / batch_count, avg_component_losses

    def train_stage_adaptive(self, stage: int, max_epochs: int, min_epochs: int):
        """
        Train a stage with adaptive termination based on curriculum learning

        Args:
            stage: Stage number (1, 2, 3)
            max_epochs: Maximum epochs for this stage
            min_epochs: Minimum epochs before considering transition
        """
        print("=" * 60)
        print(f"STAGE {stage}: Adaptive Training with Dynamic Curriculum")
        print("=" * 60)

        self.current_stage = stage
        self.stage_start_time = time.time()

        # Only reset if not resuming
        if not hasattr(self, "epoch_times") or self.epoch_times is None:
            self.epoch_times = []

        start_epoch = int(self.stage_epoch or 0)

        # Set parameter gradients for current stage
        self._configure_stage_parameters(stage)

        mode_name = f"stage{stage}"

        for epoch in range(start_epoch, max_epochs):
            epoch_start_time = time.time()
            self.stage_epoch = epoch
            self.global_epoch += 1

            # Update loss weights based on curriculum
            self._update_loss_weights_for_curriculum(stage, epoch, max_epochs)

            print(
                f"\nStage {stage} - Epoch {epoch+1}/{max_epochs} (Global: {self.global_epoch})"
            )

            # Training and validation
            train_loss, train_components = self._train_epoch(mode_name)
            val_loss, val_components = self._validate(mode_name)

            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            if len(self.epoch_times) > 10:
                self.epoch_times.pop(0)

            # Update curriculum state
            self.curriculum_state.update_loss_history(mode_name, val_loss)
            self.curriculum_state.update_component_losses(val_components)

            # Store training history
            self.history[mode_name]["train_loss"].append(train_loss)
            self.history[mode_name]["val_loss"].append(val_loss)
            self.history[mode_name]["component_losses"].append(val_components)

            # Update learning rate
            if stage == 1:
                self.scheduler_2d.step()
                current_lr = self.optimizer_2d.param_groups[0]['lr']
            elif stage == 2:
                self.scheduler_dvx.step()
                current_lr = self.optimizer_dvx.param_groups[0]['lr']
            else:
                self.scheduler_full.step()
                current_lr = self.optimizer_full.param_groups[0]['lr']
            
            # Log learning rate every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Learning rate at epoch {epoch + 1}: {current_lr:.6f}")

            # Display comprehensive results
            self._display_epoch_results(
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                train_components,
                val_components,
                epoch_time,
            )

            # Check for adaptive stage transition
            if epoch >= min_epochs:
                should_transition = self.curriculum_state.should_transition(stage)
                if should_transition:
                    print(
                        f"\n🔄 ADAPTIVE TRANSITION: Stage {stage} converged after {epoch+1} epochs"
                    )
                    print(
                        "   Detected performance plateau - transitioning to next stage"
                    )

                    self.history["stage_transitions"].append(
                        {
                            "from_stage": stage,
                            "epoch": epoch + 1,
                            "global_epoch": self.global_epoch,
                            "reason": "performance_plateau",
                        }
                    )

                    self.history["curriculum_events"].append(
                        {
                            "type": "stage_transition",
                            "stage": stage,
                            "epoch": self.global_epoch,
                            "details": f"Converged after {epoch+1} epochs",
                        }
                    )
                    break

            # Save rolling checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_rolling_checkpoint()

        print(f"\nStage {stage} completed after {epoch+1} epochs")

    def _configure_stage_parameters(self, stage: int):
        """Configure which parameters require gradients for each stage"""
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        if stage == 1:
            # Stage 1: Segmentation + Attributes (2D only)
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.seg_head.parameters():
                param.requires_grad = True
            for param in self.model.attr_head.parameters():
                param.requires_grad = True
            for param in self.model.sdf_head.parameters():
                param.requires_grad = True

        elif stage == 2:
            # Stage 2: DVX training (polygon fitting) - keep encoder frozen initially
            for param in self.model.dvx.parameters():
                param.requires_grad = True
            # Optionally unfreeze encoder in later epochs
            if self.stage_epoch > 10:
                for param in self.model.encoder.parameters():
                    param.requires_grad = True

        else:  # stage == 3
            # Stage 3: End-to-end fine-tuning (all parameters)
            for param in self.model.parameters():
                param.requires_grad = True

    def _display_epoch_results(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        train_components: Dict,
        val_components: Dict,
        epoch_time: float,
    ):
        """Display comprehensive epoch results with curriculum information"""
        eta_str = self._get_eta_string(epoch, total_epochs)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s, {eta_str}")

        # Show significant component losses
        significant_components = {
            k: v
            for k, v in val_components.items()
            if v > 0.01
            and k
            in [
                "seg",
                "dice",
                "polygon",
                "voxel",
                "topology",
                "latent_consistency",
                "graph",
            ]
        }
        if significant_components:
            comp_str = ", ".join(
                [f"{k}: {v:.3f}" for k, v in significant_components.items()]
            )
            print(f"Components: {comp_str}")

        # Show current loss weights for active components
        active_weights = {k: v for k, v in self.loss_fn.weights.items() if v > 0.001}
        if active_weights:
            weight_str = ", ".join([f"{k}: {v:.3f}" for k, v in active_weights.items()])
            print(f"Weights: {weight_str}")

        # Show curriculum status
        plateau_epochs = self.curriculum_state.epochs_without_improvement
        if plateau_epochs > 0:
            print(f"Plateau: {plateau_epochs} epochs without improvement")

    def _save_rolling_checkpoint(self):
        """Enhanced checkpoint saving with curriculum state, RNG state, and scaler state"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_2d_state_dict": self.optimizer_2d.state_dict(),
            "optimizer_dvx_state_dict": self.optimizer_dvx.state_dict(),
            "optimizer_full_state_dict": self.optimizer_full.state_dict(),
            "scheduler_2d_state_dict": self.scheduler_2d.state_dict(),
            "scheduler_dvx_state_dict": self.scheduler_dvx.state_dict(),
            "scheduler_full_state_dict": self.scheduler_full.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),  # Add AMP scaler state
            "loss_fn_state": {
                "weights": self.loss_fn.weights,
                "initial_weights": self.loss_fn.initial_weights,
            },
            "history": self.history,
            "config": self.config,
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
            "global_epoch": self.global_epoch,
            "stage_epoch": self.stage_epoch,
            "epoch_times": self.epoch_times,
            "step_counter": self._step,  # Save step counter for DVX gating
            "curriculum_state": {
                "loss_history": dict(self.curriculum_state.loss_history),
                "component_losses": dict(self.curriculum_state.component_losses),
                "epochs_without_improvement": self.curriculum_state.epochs_without_improvement,
                "best_val_loss": self.curriculum_state.best_val_loss,
                "stage_transition_epochs": self.curriculum_state.stage_transition_epochs,
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        checkpoint_path = self.ROLLING_CHECKPOINT
        torch.save(checkpoint, checkpoint_path)
        print(f"Rolling checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename):
        """
        Enhanced checkpoint loading with architecture compatibility handling
        Safely handles model architecture changes by filtering incompatible parameters
        """
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)

        # === SAFE MODEL STATE LOADING ===
        model_state = checkpoint["model_state_dict"]
        current_model_keys = set(self.model.state_dict().keys())
        
        # Filter parameters into compatible and incompatible
        compatible_state = {}
        incompatible_keys = []
        missing_keys = []
        
        # Check each parameter in the checkpoint
        for key, value in model_state.items():
            if key in current_model_keys:
                # Check if tensor shapes match
                current_param = self.model.state_dict()[key]
                if current_param.shape == value.shape:
                    compatible_state[key] = value
                else:
                    incompatible_keys.append(f"{key} (shape mismatch: {value.shape} -> {current_param.shape})")
            else:
                incompatible_keys.append(f"{key} (parameter not found in current model)")
        
        # Check for missing parameters in checkpoint
        for key in current_model_keys:
            if key not in model_state:
                missing_keys.append(key)
        
        # Load compatible parameters only
        loaded_keys, unexpected_keys = self.model.load_state_dict(compatible_state, strict=False)
        
        # Report parameter loading status
        print(f"✓ Successfully loaded {len(compatible_state)} compatible parameters")
        
        if incompatible_keys:
            print(f"⚠ Skipped {len(incompatible_keys)} incompatible parameters:")
            for key in incompatible_keys[:10]:  # Show first 10
                print(f"    - {key}")
            if len(incompatible_keys) > 10:
                print(f"    ... and {len(incompatible_keys) - 10} more")
        
        if missing_keys:
            print(f"⚠ {len(missing_keys)} parameters will use random initialization:")
            for key in missing_keys[:10]:  # Show first 10
                print(f"    - {key}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys) - 10} more")

        # === OPTIMIZER STATES LOADING ===
        try:
            self.optimizer_2d.load_state_dict(checkpoint["optimizer_2d_state_dict"])
            print("✓ Loaded optimizer_2d state")
        except Exception as e:
            print(f"⚠ Could not load optimizer_2d state: {e}")

        try:
            self.optimizer_dvx.load_state_dict(checkpoint["optimizer_dvx_state_dict"])
            print("✓ Loaded optimizer_dvx state")
        except Exception as e:
            print(f"⚠ Could not load optimizer_dvx state: {e}")

        try:
            self.optimizer_full.load_state_dict(checkpoint["optimizer_full_state_dict"])
            print("✓ Loaded optimizer_full state")
        except Exception as e:
            print(f"⚠ Could not load optimizer_full state: {e}")

        # === AMP SCALER STATE ===
        if "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print("✓ Loaded AMP scaler state")
            except Exception as e:
                print(f"⚠ Could not load scaler state: {e}")

        # === SCHEDULER STATES ===
        scheduler_mappings = [
            ("scheduler_2d_state_dict", self.scheduler_2d),
            ("scheduler_dvx_state_dict", self.scheduler_dvx),
            ("scheduler_full_state_dict", self.scheduler_full),
        ]
        
        for state_key, scheduler_obj in scheduler_mappings:
            if state_key in checkpoint:
                try:
                    scheduler_obj.load_state_dict(checkpoint[state_key])
                    print(f"✓ Loaded {state_key.replace('_state_dict', '')} scheduler")
                except Exception as e:
                    print(f"⚠ Could not load {state_key}: {e}")

        # === LOSS FUNCTION STATE ===
        if "loss_fn_state" in checkpoint:
            try:
                loaded_weights = checkpoint["loss_fn_state"]["weights"]
                if isinstance(loaded_weights, dict):
                    # Handle device transfer for tensor weights
                    self.loss_fn.weights = {
                        k: (v.to(self.device) if torch.is_tensor(v) else v)
                        for k, v in loaded_weights.items()
                    }
                else:
                    self.loss_fn.weights = loaded_weights
                
                self.loss_fn.initial_weights = checkpoint["loss_fn_state"]["initial_weights"]
                print("✓ Loaded loss function weights")
            except Exception as e:
                print(f"⚠ Could not load loss function state: {e}")

        # === TRAINING HISTORY ===
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            print("✓ Loaded training history")

        # === TRAINING STATE VARIABLES ===
        state_variables = [
            ("current_stage", "current_stage"),
            ("current_epoch", "current_epoch"), 
            ("global_epoch", "global_epoch"),
            ("stage_epoch", "stage_epoch"),
            ("epoch_times", "epoch_times"),
            ("step_counter", "_step"),
        ]
        
        for checkpoint_key, attr_name in state_variables:
            if checkpoint_key in checkpoint:
                setattr(self, attr_name, checkpoint[checkpoint_key])
                print(f"✓ Restored {checkpoint_key}: {getattr(self, attr_name)}")

        # === CURRICULUM STATE RESTORATION ===
        if "curriculum_state" in checkpoint:
            try:
                cs = checkpoint["curriculum_state"]
                
                # Restore loss history deques
                for key, history in cs.get("loss_history", {}).items():
                    self.curriculum_state.loss_history[key] = deque(
                        history, maxlen=self.config.curriculum.plateau_detection_window * 2
                    )
                
                # Restore component loss deques
                for key, history in cs.get("component_losses", {}).items():
                    self.curriculum_state.component_losses[key] = deque(history, maxlen=20)
                
                # Restore curriculum metrics
                self.curriculum_state.epochs_without_improvement = cs.get("epochs_without_improvement", 0)
                self.curriculum_state.best_val_loss = cs.get("best_val_loss", float("inf"))
                self.curriculum_state.stage_transition_epochs = cs.get("stage_transition_epochs", [])
                
                print("✓ Restored curriculum learning state")
            except Exception as e:
                print(f"⚠ Could not restore curriculum state: {e}")

        # === RNG STATE RESTORATION ===
        if "rng_state" in checkpoint:
            try:
                rs = checkpoint["rng_state"]

                # Torch RNG (CPU)
                if "torch" in rs and rs["torch"] is not None:
                    torch_state = rs["torch"]
                    if torch.is_tensor(torch_state):
                        if torch_state.dtype == torch.uint8:
                            torch.set_rng_state(torch_state)
                        else:
                            torch.set_rng_state(torch_state.byte())
                    else:
                        torch.set_rng_state(torch.ByteTensor(torch_state))
                        
                # CUDA RNG (all devices)
                if "cuda" in rs and rs["cuda"] is not None and torch.cuda.is_available():
                    cuda_state = rs["cuda"]
                    cuda_tensors = []
                    for s in cuda_state:
                        if torch.is_tensor(s):
                            cuda_tensors.append(s.byte() if s.dtype != torch.uint8 else s)
                        else:
                            cuda_tensors.append(torch.ByteTensor(s))
                    torch.cuda.set_rng_state_all(cuda_tensors)

                # NumPy RNG
                if "numpy" in rs and rs["numpy"] is not None:
                    np.random.set_state(rs["numpy"])

                # Python random RNG
                if "python" in rs and rs["python"] is not None:
                    random.setstate(rs["python"])

                print("✓ Restored RNG states for reproducibility")
            except Exception as e:
                print(f"⚠ Could not restore RNG states: {e}")

        # === DATALOADER STATE (if available) ===
        if "dataloader_state" in checkpoint:
            try:
                dl_state = checkpoint["dataloader_state"]
                if (dl_state.get("train_sampler_state") is not None and 
                    hasattr(self.train_loader.sampler, "__dict__")):
                    self.train_loader.sampler.__dict__.update(dl_state["train_sampler_state"])
                
                if (dl_state.get("val_sampler_state") is not None and 
                    hasattr(self.val_loader.sampler, "__dict__")):
                    self.val_loader.sampler.__dict__.update(dl_state["val_sampler_state"])
                
                print("✓ Restored dataloader sampler states")
            except Exception as e:
                print(f"⚠ Could not restore dataloader states: {e}")

        # === FINAL REPORT ===
        print("\n" + "="*60)
        print("CHECKPOINT LOADING SUMMARY")
        print("="*60)
        print(f"✓ Checkpoint loaded: {filename}")
        print(f"✓ Resuming from Stage {self.current_stage}, Global Epoch {self.global_epoch}")
        print(f"✓ Model parameters: {len(compatible_state)}/{len(model_state)} loaded successfully")
        
        if hasattr(self, 'curriculum_state'):
            print(f"✓ Curriculum state: {self.curriculum_state.epochs_without_improvement} epochs without improvement")
        
        if incompatible_keys:
            print(f"⚠ Architecture changes detected: {len(incompatible_keys)} parameters skipped")
            print("  This is normal after model architecture updates.")
        
        if missing_keys:
            print(f"⚠ New parameters detected: {len(missing_keys)} will use random initialization")
            print("  These will be learned quickly during resumed training.")
        
        print("="*60)
        print("Ready to resume adaptive multi-stage training!")
        print("="*60)

    def train_all_stages(self):
        """
        Run complete adaptive multi-stage training pipeline

        This is the main entry point that orchestrates the dynamic curriculum learning
        """
        if Path(self.ROLLING_CHECKPOINT).exists():
            print(f"Found existing checkpoint: {self.ROLLING_CHECKPOINT}")
            print("Resuming adaptive training from checkpoint...")
            self.load_checkpoint(self.ROLLING_CHECKPOINT)
        else:
            print("Starting fresh adaptive training pipeline...")
            self.current_stage = 1
            self.current_epoch = 0
            self.global_epoch = 0

        print("\n" + "=" * 80)
        print("ADAPTIVE MULTI-STAGE TRAINING WITH DYNAMIC CURRICULUM")
        print("Novel Training Strategies:")
        print("• Adaptive Stage Transitioning (Dynamic Curriculum)")
        print("• Topology-aware Loss Scheduling")
        print("• Multi-objective Optimization with Dynamic Weighting")
        print("• Cross-modal Latent Consistency Learning")
        print("• Graph-based Topology Constraints")
        print("=" * 80)

        # Stage 1: Adaptive 2D training
        if self.current_stage <= 1:
            print("\n🚀 STAGE 1: Adaptive 2D Segmentation + Attributes Training")
            self.train_stage_adaptive(
                stage=1,
                max_epochs=self.config.max_stage1_epochs,
                min_epochs=self.config.min_stage1_epochs,
            )
            self.current_stage = 2
            self.stage_epoch = 0
            print("\nStage 1 completed. Transitioning to Stage 2...")

        # Stage 2: Adaptive DVX training
        if self.current_stage <= 2:
            print("\n🔄 STAGE 2: Adaptive DVX Polygon Fitting Training")
            self.train_stage_adaptive(
                stage=2,
                max_epochs=self.config.max_stage2_epochs,
                min_epochs=self.config.min_stage2_epochs,
            )
            self.current_stage = 3
            self.stage_epoch = 0
            print("\nStage 2 completed. Transitioning to Stage 3...")

        # Stage 3: Adaptive end-to-end fine-tuning
        if self.current_stage <= 3:
            print("\n🎯 STAGE 3: Adaptive End-to-End Fine-tuning with Full Loss Suite")
            self.train_stage_adaptive(
                stage=3,
                max_epochs=self.config.max_stage3_epochs,
                min_epochs=self.config.min_stage3_epochs,
            )
            print("\nStage 3 completed!")

        print("\n" + "=" * 80)
        print("✅ ALL ADAPTIVE TRAINING STAGES COMPLETED!")
        print("=" * 80)

        # Generate training report
        self._generate_training_report()

        # Save final model
        self._save_checkpoint("final_adaptive_model.pth")

        # Clean up rolling checkpoint
        if Path(self.ROLLING_CHECKPOINT).exists():
            Path(self.ROLLING_CHECKPOINT).unlink()
            print(f"Cleaned up rolling checkpoint: {self.ROLLING_CHECKPOINT}")

        return self.history

    def _generate_training_report(self):
        """Generate comprehensive training report with curriculum insights"""
        print("\n" + "=" * 60)
        print("ADAPTIVE TRAINING REPORT")
        print("=" * 60)

        # Stage transition summary
        if self.history["stage_transitions"]:
            print("\n📊 Stage Transitions:")
            for transition in self.history["stage_transitions"]:
                print(
                    f"  • Stage {transition['from_stage']} → {transition['from_stage']+1}: "
                    f"Epoch {transition['epoch']} (Global: {transition['global_epoch']})"
                )
                print(f"    Reason: {transition['reason']}")

        # Dynamic weight evolution
        if self.history["dynamic_weights"]:
            print(
                f"\n⚖️  Dynamic Weight Updates: {len(self.history['dynamic_weights'])} updates"
            )
            final_weights = self.history["dynamic_weights"][-1]["weights"]
            print("  Final loss weights:")
            for name, weight in final_weights.items():
                if weight > 0.001:
                    print(f"    {name}: {weight:.3f}")

        # Curriculum events
        if self.history["curriculum_events"]:
            print(
                f"\n🎯 Curriculum Events: {len(self.history['curriculum_events'])} events"
            )
            for event in self.history["curriculum_events"][-5:]:  # Show last 5 events
                print(
                    f"  • {event['type']} at global epoch {event['epoch']}: {event['details']}"
                )

        # Performance summary
        print("\n📈 Final Performance:")
        for stage_name, data in self.history.items():
            if isinstance(data, dict) and "val_loss" in data and data["val_loss"]:
                final_loss = data["val_loss"][-1]
                best_loss = min(data["val_loss"])
                print(
                    f"  • {stage_name.upper()}: Final={final_loss:.4f}, Best={best_loss:.4f}"
                )

        print("\n🏁 Training completed with novel adaptive curriculum strategies!")
        print("=" * 60)

    def _save_checkpoint(self, filename):
        """Save final training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_2d_state_dict": self.optimizer_2d.state_dict(),
            "optimizer_dvx_state_dict": self.optimizer_dvx.state_dict(),
            "optimizer_full_state_dict": self.optimizer_full.state_dict(),
            "scheduler_2d_state_dict": self.scheduler_2d.state_dict(),
            "scheduler_dvx_state_dict": self.scheduler_dvx.state_dict(),
            "scheduler_full_state_dict": self.scheduler_full.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss_fn_state": {
                "weights": self.loss_fn.weights,
                "initial_weights": self.loss_fn.initial_weights,
            },
            "history": self.history,
            "config": self.config,
            "final_stage": self.current_stage,
            "total_epochs": self.global_epoch,
            "training_complete": True,
            "curriculum_summary": {
                "stage_transitions": len(self.history["stage_transitions"]),
                "weight_updates": len(self.history["dynamic_weights"]),
                "curriculum_events": len(self.history["curriculum_events"]),
            },
        }
        torch.save(checkpoint, filename)
        print(f"Final model saved: {filename}")


# Legacy compatibility class
class MultiStageTrainer(AdaptiveMultiStageTrainer):
    """
    Legacy wrapper for backward compatibility
    Redirects to the new adaptive trainer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Note: Using enhanced AdaptiveMultiStageTrainer with dynamic curriculum")

    def train_stage1(self, epochs=None):
        """Legacy method - redirects to adaptive training"""
        max_epochs = epochs or self.config.max_stage1_epochs
        min_epochs = self.config.min_stage1_epochs
        return self.train_stage_adaptive(1, max_epochs, min_epochs)

    def train_stage2(self, epochs=None):
        """Legacy method - redirects to adaptive training"""
        max_epochs = epochs or self.config.max_stage2_epochs
        min_epochs = self.config.min_stage2_epochs
        return self.train_stage_adaptive(2, max_epochs, min_epochs)

    def train_stage3(self, epochs=None):
        """Legacy method - redirects to adaptive training"""
        max_epochs = epochs or self.config.max_stage3_epochs
        min_epochs = self.config.min_stage3_epochs
        return self.train_stage_adaptive(3, max_epochs, min_epochs)