
import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'../../my_models') # including the path to my_models folder
from constants import RAUG_PATH
from my_model import set_model
sys.path.insert(0,RAUG_PATH)
from raug.checkpoints import save_model_as_onnx, load_model
import torch


################################################################################
# Models configuration
################################################################################
_LABELS_NAME = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
_MODEL_NAME = 'resnet-50'
_NEURONS_REDUCER_BLOCK = 0
_COMB_METHOD = "metablock"
_COMB_CONFIG = [64, 45]
_CHECKPOINT_PATH = "./results/metablock_resnet-50_fold_3_17066400974411047/best-checkpoint/best-checkpoint.pth"
_FOLDER_PATH = "./results"
################################################################################


# Loading the model
model = set_model(_MODEL_NAME, len(_LABELS_NAME), neurons_reducer_block=_NEURONS_REDUCER_BLOCK,
                      comb_method=_COMB_METHOD, comb_config=_COMB_CONFIG, pretrained=False)

model = load_model(_CHECKPOINT_PATH, model)
model.eval()

print("- Saving ONNX model...")

dummy_img_input = torch.randn(1, 3, 224, 224)
dummy_meta_input = torch.randn(1, _COMB_CONFIG[1])
dummy_meta_input = dummy_meta_input.type(torch.LongTensor)

print(dummy_img_input.type(), dummy_img_input.shape)
print(dummy_meta_input.type(), dummy_meta_input.shape)


_MODEL_NAME = _MODEL_NAME + f"-{_COMB_CONFIG[1]}-meta.onnx"

save_model_as_onnx(model, _FOLDER_PATH, _MODEL_NAME,
                    (dummy_img_input, dummy_meta_input),
                    ["img_input", "meta_data_input"],
                    ["output_prob"],
                    {"img_input": {0: "batch_size"}, "meta_data_input": {0: "batch_size"}})