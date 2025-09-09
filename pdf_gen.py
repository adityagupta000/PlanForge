# import os
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.units import mm
# from reportlab.pdfgen import canvas


# def get_code_files(directory, excluded_files=None, excluded_dirs=None):
#     """Fetch all .py files from the given directory (excluding sensitive files and dirs)."""
#     if excluded_files is None:
#         excluded_files = {
#             ".DS_Store",
#             "Thumbs.db",
#             "Desktop.ini",
#             "__init__.py", 
#             "pdf_gen.py",  
#         }

#     if excluded_dirs is None:
#         excluded_dirs = {
#             "node_modules",
#             ".git",
#             "__pycache__",
#             "build",
#             "dist",
#             "logs",
#             "venv",
#             "env",
#         }

#     code_files = {}

#     for root, dirs, files in os.walk(directory):
#         # Skip excluded directories
#         dirs[:] = [d for d in dirs if d not in excluded_dirs]

#         for file in files:
#             # Skip excluded files
#             if file in excluded_files:
#                 continue

#             # Only include Python files
#             if not file.endswith(".py"):
#                 continue

#             file_path = os.path.join(root, file)

#             try:
#                 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code_files[file_path] = f.readlines()
#             except Exception as e:
#                 print(f"❌ Error reading {file_path}: {e}")
#                 code_files[file_path] = [f"[Error reading file: {str(e)}]"]

#     return code_files


# def create_pdf(code_data, output_pdf="Python_Code_Export.pdf"):
#     c = canvas.Canvas(output_pdf, pagesize=A4)
#     width, height = A4
#     margin = 20 * mm
#     line_height = 10
#     y = height - margin

#     # Title
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(margin, y, "🐍 Python Project Code Export")
#     y -= 3 * line_height

#     file_paths = sorted(list(code_data.keys()))

#     # File list
#     c.setFont("Courier", 8)
#     for path in file_paths:
#         if y < margin:
#             c.showPage()
#             c.setFont("Courier", 8)
#             y = height - margin
#         display_path = os.path.relpath(path)
#         c.drawString(margin, y, f"- [PY] {display_path}")
#         y -= line_height

#     # Page break before code content
#     c.showPage()
#     y = height - margin

#     # File contents
#     for file_path in file_paths:
#         lines = code_data[file_path]
#         print(f"📄 Adding: {file_path}")

#         if y < margin + 3 * line_height:
#             c.showPage()
#             y = height - margin

#         # File header
#         rel_path = os.path.relpath(file_path)
#         c.setFont("Helvetica-Bold", 12)
#         c.drawString(margin, y, f"📄 File: {rel_path}")
#         y -= line_height

#         # Separator
#         c.setFont("Courier", 8)
#         c.drawString(margin, y, "=" * 80)
#         y -= line_height

#         # File content
#         for line_num, line in enumerate(lines, 1):
#             if y < margin:
#                 c.showPage()
#                 c.setFont("Courier", 8)
#                 y = height - margin

#             line = line.strip("\n").encode("latin-1", "replace").decode("latin-1")
#             display_line = f"{line_num:3d}: {line[:280]}"
#             c.drawString(margin, y, display_line)
#             y -= line_height

#         # Spacing
#         y -= line_height
#         if y > margin:
#             c.drawString(margin, y, "-" * 80)
#             y -= 2 * line_height

#     c.save()
#     print(f"✅ PDF successfully created: {output_pdf}")
#     print(f"📊 Total files processed: {len(code_data)}")


# def main():
#     root_dir = os.path.dirname(os.path.abspath(__file__))

#     print("🔍 Scanning for Python files...")
#     code_files = get_code_files(root_dir)

#     if not code_files:
#         print("❌ No Python files found to process!")
#         return

#     print(f"📁 Found {len(code_files)} Python files to include in PDF")
#     for file_path in sorted(code_files.keys()):
#         print(f"   📄 {os.path.relpath(file_path)}")

#     create_pdf(code_files)


# if __name__ == "__main__":
#     main()

import os
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ✅ Only include the files that are part of training
INCLUDED_FILES = {
    "train.py",
    "config.py",
    "dataset.py",
    os.path.join("models", "model.py"),
    os.path.join("models", "encoder.py"),
    os.path.join("models", "heads.py"),
    os.path.join("models", "dvx.py"),
    os.path.join("models", "extrusion.py"),
    os.path.join("training", "trainer.py"),
    os.path.join("training", "losses.py"),
    os.path.join("utils", "visualization.py"),
}

def get_code_files(directory):
    """Fetch only the included .py files from the project."""
    code_files = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue

            rel_path = os.path.relpath(os.path.join(root, file), directory)

            # ✅ Only process whitelisted files
            if rel_path not in INCLUDED_FILES:
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_files[file_path] = f.readlines()
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                code_files[file_path] = [f"[Error reading file: {str(e)}]"]

    return code_files


def wrap_long_line(line, max_width=100, subsequent_indent="    "):
    """Wrap a long line of code, preserving indentation."""
    # Preserve original indentation
    stripped_line = line.lstrip()
    original_indent = line[:len(line) - len(stripped_line)]
    
    if len(line) <= max_width:
        return [line]
    
    # Use textwrap but preserve the original indentation
    wrapped = textwrap.wrap(
        stripped_line,
        width=max_width - len(original_indent),
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False
    )
    
    # Add back the original indentation to all lines
    result = []
    for i, wrapped_line in enumerate(wrapped):
        if i == 0:
            result.append(original_indent + wrapped_line)
        else:
            result.append(original_indent + subsequent_indent + wrapped_line)
    
    return result


def create_pdf(code_data, output_pdf="Training_Code_Export.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    line_height = 10
    y = height - margin
    
    # Calculate max characters that fit on a line with Courier 8pt font
    # Approximately 100 characters fit comfortably on A4 with margins
    max_line_width = 100

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "🐍 Training Code Export (Complete Code - No Truncation)")
    y -= 3 * line_height

    file_paths = sorted(list(code_data.keys()))

    # File list
    c.setFont("Courier", 8)
    for path in file_paths:
        if y < margin:
            c.showPage()
            c.setFont("Courier", 8)
            y = height - margin
        display_path = os.path.relpath(path)
        c.drawString(margin, y, f"- [PY] {display_path}")
        y -= line_height

    # Page break before code content
    c.showPage()
    y = height - margin

    # File contents
    for file_path in file_paths:
        lines = code_data[file_path]
        print(f"📄 Adding: {file_path}")

        if y < margin + 3 * line_height:
            c.showPage()
            y = height - margin

        # File header
        rel_path = os.path.relpath(file_path)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"📄 File: {rel_path}")
        y -= line_height

        # Separator
        c.setFont("Courier", 8)
        c.drawString(margin, y, "=" * 80)
        y -= line_height

        # File content with proper line wrapping
        for line_num, original_line in enumerate(lines, 1):
            # Clean the line for PDF encoding
            clean_line = original_line.rstrip("\n\r")
            clean_line = clean_line.encode("latin-1", "replace").decode("latin-1")
            
            # Wrap long lines
            wrapped_lines = wrap_long_line(clean_line, max_line_width)
            
            for i, wrapped_line in enumerate(wrapped_lines):
                if y < margin:
                    c.showPage()
                    c.setFont("Courier", 8)
                    y = height - margin

                # Show line number only for the first wrapped segment
                if i == 0:
                    display_line = f"{line_num:3d}: {wrapped_line}"
                else:
                    display_line = f"    + {wrapped_line}"  # Continuation indicator
                
                c.drawString(margin, y, display_line)
                y -= line_height

        # Spacing between files
        y -= line_height
        if y > margin:
            c.drawString(margin, y, "-" * 80)
            y -= 2 * line_height

    c.save()
    print(f"✅ PDF successfully created: {output_pdf}")
    print(f"📊 Total files processed: {len(code_data)}")
    print("📝 Note: Long lines have been wrapped to prevent truncation")


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    print("🔍 Scanning for included Python files...")
    code_files = get_code_files(root_dir)

    if not code_files:
        print("❌ No included Python files found to process!")
        return

    print(f"📁 Found {len(code_files)} Python files to include in PDF")
    for file_path in sorted(code_files.keys()):
        print(f"   📄 {os.path.relpath(file_path)}")

    create_pdf(code_files)


if __name__ == "__main__":
    main()