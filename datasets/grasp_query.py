# from lib.utils.misc import ImmutableClass


class ImmutableClass(type):

    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")
    
class Queries(metaclass=ImmutableClass):
    # 1. obj canonical space
    OBJ_VERTS_OBJ = "obj_verts_obj"  # = obj_verts_can
    OBJ_NORMALS_OBJ = "obj_normals_obj"

    # OBJ_VERTS_OBJ_DS = "obj_verts_obj_ds"
    OBJ_VERTS_OBJ_DS = "pos"
    OBJ_VERTS_OBJ_DS_ORI = "pos_ori"
    OBJ_NORMALS_OBJ_DS = "obj_normals_obj_ds"
    OBJ_BPS = "obj_bps"
    CORNERS_OBJ = "corners_obj"  # = corners_can
    HAND_VERTS_OBJ = "hand_verts_obj"
    HAND_TRANSF = "hand_transf"
    JOINTS_OBJ = "joints_obj"
    # ROOT_JOINT_OBJ = "root_joint_obj"
    HAND_TRANSL_OBJ = "hand_transl_obj"
    # HAND_POSE_OBJ = "hand_pose_obj"
    HAND_POSE_OBJ = "hand_pose_obj"
    CONTACTNESS = 'contactness'
    VERTEX_CONTACT = 'vertex_contact'
    ANCHOR_ELASTI = 'anchor_elasti'
    O2H_SIGNED_DISTANCE = 'o2h_signed_distance'
    H2O_SIGNED_DISTANCE = 'h2o_signed_distance'
    CONTACT_REGION = 'contact_region'
    CONTACT_REGION_ID = 'contact_region_id'
    HAND_PARA = 'x'

    # misc:
    HAND_SHAPE = "hand_shape"
    SAMPLE_PATH = "sample_path"
    SAMPLE_IDX = "sample_idx"
    SAMPLE_IDENTIFIER = "sample_identifier"
    OBJ_FACES = "obj_faces"
    OBJ_ROTMAT = "obj_rotmat"
    HAND_FACES = "hand_faces"
    SIDE = "side"
    OBJ_VERTS_PADDING_MASK = "obj_verts_padding_mask"
    OBJ_FACES_PADDING_MASK = "obj_face_padding_mask"

    OBJ_ID = "obj_id"
    OBJ_PATH = "obj_path"

    INTENT_ID = "intent_id"
    INTENT_NAME = "intent_name"
    INTENT_VEC = "intent_vec"
    ALT_JOINTS_OBJ = "alt_joints_obj"
    ALT_HAND_POSE_OBJ = "alt_hand_pose_obj"
    ALT_HAND_SHAPE = "alt_hand_shape"
    ALT_HAND_VERTS_OBJ = "alt_hand_verts_obj"
    ALT_HAND_TRANSL_OBJ = "alt_hand_transl_obj"
    GUIDENCE = "guidence"
    INS_DISC = "ins_disc"
    INS_MASK_DISC = "ins_mask_disc"
    TASK_DISC = "task_disc"
    TASK_MASK_DISC = "task_mask_disc"
    OBJ_DISC = "obj_disc"
    OBJ_MASK_DISC = "obj_mask_disc"
    VIEWPOINT = "viewpoint"
    INS_PATH = "ins_path"


def match_collate_queries(query_spin):
    object_vertex_queries = [
        Queries.OBJ_VERTS_OBJ,
        Queries.OBJ_NORMALS_OBJ,
    ]
    object_face_quries = [
        Queries.OBJ_FACES,
    ]

    if query_spin in object_vertex_queries:
        return Queries.OBJ_VERTS_PADDING_MASK
    elif query_spin in object_face_quries:
        return Queries.OBJ_FACES_PADDING_MASK
