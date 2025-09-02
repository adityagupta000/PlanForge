"""
Multi-stage training system for the Neural-Geometric 3D Model Generator
Enhanced with progress bars, ETA, auto-resume, and rolling checkpoints
"""

import torch
import torch.nn.utils
import time
from pathlib import Path
from tqdm import tqdm

from .losses import ResearchGradeLoss
from config import DEFAULT_TRAINING_CONFIG, DEFAULT_LOSS_CONFIG


class MultiStageTrainer:
    """
    Multi-stage training following the research approach:
    Stage 1: Segmentation + Attributes (2D only)
    Stage 2: DVX training (polygon fitting)
    Stage 3: End-to-end fine-tuning (all losses)
    
    Enhanced with:
    - Progress bars with tqdm
    - ETA calculation and display
    - Auto-resume functionality
    - Rolling checkpoint system
    """

    def __init__(self, model, train_loader, val_loader, device=None, config=None):
        if config is None:
            config = DEFAULT_TRAINING_CONFIG
        
        self.model = model.to(device or config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.device
        self.config = config

        # Training state tracking for resume functionality
        self.current_stage = 1
        self.current_epoch = 0
        self.stage_start_time = None
        self.epoch_times = []  # Track epoch durations for ETA calculation

        # Different optimizers for different stages
        self.optimizer_2d = torch.optim.AdamW(
            list(self.model.encoder.parameters())
            + list(self.model.seg_head.parameters())
            + list(self.model.attr_head.parameters())
            + list(self.model.sdf_head.parameters()),
            lr=config.stage1_lr,
            weight_decay=config.stage1_weight_decay,
        )

        self.optimizer_dvx = torch.optim.AdamW(
            self.model.dvx.parameters(), 
            lr=config.stage2_lr, 
            weight_decay=config.stage2_weight_decay
        )

        self.optimizer_full = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.stage3_lr, 
            weight_decay=config.stage3_weight_decay
        )

        # Learning rate schedulers
        self.scheduler_2d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_2d, T_max=config.stage1_epochs
        )
        self.scheduler_dvx = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dvx, T_max=config.stage2_epochs
        )
        self.scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_full, T_max=config.stage3_epochs
        )

        # Loss functions for different stages
        self.loss_2d = ResearchGradeLoss(
            seg_weight=1.0,
            dice_weight=1.0,
            sdf_weight=0.5,
            attr_weight=1.0,
            polygon_weight=0.0,
            voxel_weight=0.0,
            topology_weight=0.5,
        )

        self.loss_dvx = ResearchGradeLoss(
            seg_weight=0.1,
            dice_weight=0.0,
            sdf_weight=0.0,
            attr_weight=0.0,
            polygon_weight=1.0,
            voxel_weight=0.0,
            topology_weight=0.0,
        )

        self.loss_full = ResearchGradeLoss(**DEFAULT_LOSS_CONFIG.__dict__)

        self.history = {
            "stage1": {"train_loss": [], "val_loss": []},
            "stage2": {"train_loss": [], "val_loss": []},
            "stage3": {"train_loss": [], "val_loss": []},
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

    def _train_epoch(self, mode="stage1"):
        """Unified training epoch method for all stages"""
        self.model.train()
        total_loss = 0
        
        # Select appropriate optimizer and loss function based on mode
        if mode == "stage1":
            optimizer = self.optimizer_2d
            loss_fn = self.loss_2d
        elif mode == "stage2":
            optimizer = self.optimizer_dvx
            loss_fn = self.loss_dvx
        else:  # stage3
            optimizer = self.optimizer_full
            loss_fn = self.loss_full

        # Progress bar for training batches
        train_pbar = tqdm(
            self.train_loader, 
            desc=f"Training {mode.upper()}", 
            leave=False,
            ncols=100
        )

        for batch in train_pbar:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            # Forward pass
            predictions = self.model(batch["image"])

            # Prepare targets based on training mode
            if mode == "stage1":
                targets = {"mask": batch["mask"], "attributes": batch["attributes"]}
            elif mode == "stage2":
                targets = {
                    "polygons_gt": {
                        "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                        "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                    }
                }
            else:  # stage3
                targets = {
                    "mask": batch["mask"],
                    "attributes": batch["attributes"],
                    "voxels_gt": batch["voxels_gt"],
                    "polygons_gt": {
                        "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                        "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                    },
                }

            loss, loss_components = loss_fn(predictions, targets)

            loss.backward()
            
            # Apply gradient clipping based on mode
            if mode == "stage2":
                torch.nn.utils.clip_grad_norm_(self.model.dvx.parameters(), self.config.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        return total_loss / len(self.train_loader)

    def _validate(self, mode="stage1"):
        """Unified validation method for all stages"""
        self.model.eval()
        total_loss = 0

        # Select appropriate loss function based on mode
        if mode == "stage1":
            loss_fn = self.loss_2d
        elif mode == "stage2":
            loss_fn = self.loss_dvx
        else:  # stage3
            loss_fn = self.loss_full

        # Progress bar for validation batches
        val_pbar = tqdm(
            self.val_loader, 
            desc=f"Validating {mode.upper()}", 
            leave=False,
            ncols=100
        )

        with torch.no_grad():
            for batch in val_pbar:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                predictions = self.model(batch["image"])
                
                # Prepare targets based on validation mode
                if mode == "stage1":
                    targets = {"mask": batch["mask"], "attributes": batch["attributes"]}
                elif mode == "stage2":
                    targets = {
                        "polygons_gt": {
                            "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                            "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                        }
                    }
                else:  # stage3
                    targets = {
                        "mask": batch["mask"],
                        "attributes": batch["attributes"],
                        "voxels_gt": batch["voxels_gt"],
                        "polygons_gt": {
                            "polygons": batch["polygons_gt"]["polygons"].to(self.device),
                            "valid_mask": batch["polygons_gt"]["valid_mask"].to(self.device),
                        },
                    }

                loss, _ = loss_fn(predictions, targets)
                current_loss = loss.item()
                total_loss += current_loss
                
                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        return total_loss / len(self.val_loader)

    def train_stage1(self, epochs=None):
        """Stage 1: Train segmentation + attributes (2D supervision only)"""
        epochs = epochs or self.config.stage1_epochs
        
        print("=" * 50)
        print("STAGE 1: Segmentation + Attributes Training")
        print("=" * 50)

        self.current_stage = 1
        self.stage_start_time = time.time()
        self.epoch_times = []

        # Freeze DVX and extrusion modules
        for param in self.model.dvx.parameters():
            param.requires_grad = False
        for param in self.model.extrusion.parameters():
            param.requires_grad = False

        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Training and validation
            train_loss = self._train_epoch("stage1")
            val_loss = self._validate("stage1")

            # Record epoch time for ETA calculation
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Keep only last 10 epoch times for rolling average
            if len(self.epoch_times) > 10:
                self.epoch_times.pop(0)

            self.history["stage1"]["train_loss"].append(train_loss)
            self.history["stage1"]["val_loss"].append(val_loss)

            self.scheduler_2d.step()

            # Display results with ETA
            eta_str = self._get_eta_string(epoch, epochs)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch time: {epoch_time:.1f}s, {eta_str}")

            # Save rolling checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_rolling_checkpoint()

    def train_stage2(self, epochs=None):
        """Stage 2: Train DVX (polygon fitting)"""
        epochs = epochs or self.config.stage2_epochs
        
        print("=" * 50)
        print("STAGE 2: DVX Polygon Fitting Training")
        print("=" * 50)

        self.current_stage = 2
        self.stage_start_time = time.time()
        self.epoch_times = []

        # Freeze encoder and other heads, unfreeze DVX
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.seg_head.parameters():
            param.requires_grad = False
        for param in self.model.attr_head.parameters():
            param.requires_grad = False
        for param in self.model.sdf_head.parameters():
            param.requires_grad = False
        for param in self.model.dvx.parameters():
            param.requires_grad = True

        start_epoch = self.current_epoch 

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self._train_epoch("stage2")
            val_loss = self._validate("stage2")

            # Record epoch time for ETA calculation
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Keep only last 10 epoch times for rolling average
            if len(self.epoch_times) > 10:
                self.epoch_times.pop(0)

            self.history["stage2"]["train_loss"].append(train_loss)
            self.history["stage2"]["val_loss"].append(val_loss)

            self.scheduler_dvx.step()

            # Display results with ETA
            eta_str = self._get_eta_string(epoch, epochs)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch time: {epoch_time:.1f}s, {eta_str}")

            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_rolling_checkpoint()

    def train_stage3(self, epochs=None):
        """Stage 3: End-to-end fine-tuning with all losses"""
        epochs = epochs or self.config.stage3_epochs
        
        print("=" * 50)
        print("STAGE 3: End-to-End Fine-tuning")
        print("=" * 50)

        self.current_stage = 3
        self.stage_start_time = time.time()
        self.epoch_times = []

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        start_epoch = self.current_epoch

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self._train_epoch("stage3")
            val_loss = self._validate("stage3")

            # Record epoch time for ETA calculation
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Keep only last 10 epoch times for rolling average
            if len(self.epoch_times) > 10:
                self.epoch_times.pop(0)

            self.history["stage3"]["train_loss"].append(train_loss)
            self.history["stage3"]["val_loss"].append(val_loss)

            self.scheduler_full.step()

            # Display results with ETA
            eta_str = self._get_eta_string(epoch, epochs)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch time: {epoch_time:.1f}s, {eta_str}")

            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_rolling_checkpoint()

    def _save_rolling_checkpoint(self):
        """Save rolling checkpoint that overwrites previous one"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_2d_state_dict": self.optimizer_2d.state_dict(),
            "optimizer_dvx_state_dict": self.optimizer_dvx.state_dict(),
            "optimizer_full_state_dict": self.optimizer_full.state_dict(),
            "scheduler_2d_state_dict": self.scheduler_2d.state_dict(),
            "scheduler_dvx_state_dict": self.scheduler_dvx.state_dict(),
            "scheduler_full_state_dict": self.scheduler_full.state_dict(),
            "history": self.history,
            "config": self.config,
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
            "epoch_times": self.epoch_times,
        }
        
        checkpoint_path = "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Rolling checkpoint saved: {checkpoint_path}")

    def _save_checkpoint(self, filename):
        """Save training checkpoint (legacy method for final model)"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_2d_state_dict": self.optimizer_2d.state_dict(),
            "optimizer_dvx_state_dict": self.optimizer_dvx.state_dict(),
            "optimizer_full_state_dict": self.optimizer_full.state_dict(),
            "scheduler_2d_state_dict": self.scheduler_2d.state_dict(),
            "scheduler_dvx_state_dict": self.scheduler_dvx.state_dict(),
            "scheduler_full_state_dict": self.scheduler_full.state_dict(),
            "history": self.history,
            "config": self.config,
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Load training checkpoint with resume support"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_2d.load_state_dict(checkpoint["optimizer_2d_state_dict"])
        self.optimizer_dvx.load_state_dict(checkpoint["optimizer_dvx_state_dict"])
        self.optimizer_full.load_state_dict(checkpoint["optimizer_full_state_dict"])
        
        if "scheduler_2d_state_dict" in checkpoint:
            self.scheduler_2d.load_state_dict(checkpoint["scheduler_2d_state_dict"])
            self.scheduler_dvx.load_state_dict(checkpoint["scheduler_dvx_state_dict"])
            self.scheduler_full.load_state_dict(checkpoint["scheduler_full_state_dict"])
        
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            
        # Restore training state for resuming
        if "current_stage" in checkpoint:
            self.current_stage = checkpoint["current_stage"]
        if "current_epoch" in checkpoint:
            self.current_epoch = checkpoint["current_epoch"]
        if "epoch_times" in checkpoint:
            self.epoch_times = checkpoint["epoch_times"]
        
        print(f"Checkpoint loaded: {filename}")
        print(f"Resuming from Stage {self.current_stage}, Epoch {self.current_epoch + 1}")

    def train_all_stages(self):
        """Run complete multi-stage training pipeline with auto-resume"""
    
        checkpoint_path = "latest_checkpoint.pth"
        if Path(checkpoint_path).exists():
           print(f"Found existing checkpoint: {checkpoint_path}")
           print("Resuming training from checkpoint...")
           self.load_checkpoint(checkpoint_path)
        else:
           print("Starting fresh training pipeline...")
           self.current_stage = 1
           self.current_epoch = 0
    
        # Stage 1
        if self.current_stage <= 1:
           print("Starting/Resuming Stage 1...")
           self.train_stage1()
           print("\nStage 1 completed. Starting Stage 2...")
    
        # Stage 2
        if self.current_stage <= 2:
           print("Starting/Resuming Stage 2...")
           self.train_stage2()
           print("\nStage 2 completed. Starting Stage 3...")
    
        # Stage 3
        if self.current_stage <= 3:
           print("Starting/Resuming Stage 3...")
           self.train_stage3()
           print("\nAll training stages completed!")
    
        # Save final model
        self._save_checkpoint("final_model.pth")
    
        # Clean up rolling checkpoint
        if Path(checkpoint_path).exists():
           Path(checkpoint_path).unlink()
           print(f"Cleaned up rolling checkpoint: {checkpoint_path}")
        
        return self.history
