Third-party Modules (Compilation Required)
For modules that require CUDA compilation or specific C++ bindings (e.g., Chamfer3D, CSDF), please clone them into the thirdparty/ directory and install them manually.

Directory Structure:

Plaintext

    TOSC/
    ├── src/
    ├── thirdparty/
    │   ├── Chamfer3D/
    │   ├── CSDF/
    │   ├── V-HACD/
    │   └── ...
Installation Steps:

Chamfer3D We use Chamfer3D for distance calculation.

Bash

cd thirdparty
git clone [https://github.com/krrish94/ChamferDistance.git](https://github.com/krrish94/ChamferDistance.git) Chamfer3D
cd Chamfer3D
python setup.py install
CSDF & Libmesh Please refer to CSDF (or the specific repo you used) for mesh processing.

Bash

# Example command (adjust based on your actual usage)
cd thirdparty
git clone [https://github.com/marian42/mesh_to_sdf.git](https://github.com/marian42/mesh_to_sdf.git) CSDF
# Follow their installation guide
V-HACD Download the binary or source from [V-HACD](https://github.com/kmammou/v-hacd) and place it in thirdparty/v-hacd.