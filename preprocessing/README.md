# 📦 Data Preparation

Our dataset is constructed based on **OakInk** for object-grasp interactions and **CapGrasp** for task-oriented language instructions. Additionally, we provide a rendering pipeline to generate partial point clouds that simulate real-world occlusion.
## 1. Base Datasets
Please download the original datasets from their respective repositories and organize them in the `data/` directory.

**Grasping Data (OakInk):**
    Download the `OakInk-Shape` and `OakInk-Image` data from the [official OakInk repository](https://github.com/oakink/OakInk).

**Task Instructions (CapGrasp):**
    We utilize the language instructions provided by [CapGrasp (SemGrasp)](https://kailinli.github.io/SemGrasp/). Please download the instruction mappings relevant to the OakInk subset.


## 2. Partial Point Cloud Generation
To simulate open-world partial observations, we use Blender to render depth images and back-project them into partial point clouds.

### Prerequisites
1.  **Install Blender:** Download and install [Blender](https://blender.org/download/) (ensure the `blender` command is added to your PATH).
2.  **Install Python Bindings:** You may need to install additional libraries for handling EXR files:
    ```bash
    pip install OpenEXR Imath
    ```

#### Generation Pipeline
The code for data generation is located in the `preprocessing/` directory (or specify your folder name).

**Step 1: Prepare Object List**
Create a list of normalized 3D objects (in `.obj` format) you wish to process. We provide a template file `OakInkObjects.txt` and `OakInkVirtualObjects.txt` as a reference. Ensure your list points to the valid paths of your OakInk objects.

**Step 2: Render Depth Images**
Use the provided Blender script to render depth images from multiple viewpoints.
* `[data directory]`: Path to your object meshes.
* `[file list]`: The text file containing object names (from Step 1).
* `[output directory]`: Where to save the raw renders.
* `[num scans]`: Number of viewpoints per object.

```bash
# Syntax: blender -b -P Depth_Renderer.py [data_dir] [file_list] [output_dir] [num_scans]

blender -b -P Depth_Renderer.py [path_of_OakInk] [OakInkObjects.txt/OakInkVirtualObjects.txt] ./dump 10
```

Note: Intermediate files will be saved in OpenEXR format (*.exr). You can modify the camera intrinsics in Depth_Renderer.py, which are automatically saved to intrinsics.txt.

**Step 3: Reproject to Point Clouds**

Convert the rendered depth maps (*.exr) into partial point clouds (*.pcd) and visualization images (*.png).

```bash
python OakInkObjects.py \
    --list_file ModelNet_Flist.txt \
    --intrinsics intrinsics.txt \
    --output_dir ./dump \
    --num_scans 10
```

After these steps, your dump/ directory will contain the processed partial point clouds ready for training/inference.
