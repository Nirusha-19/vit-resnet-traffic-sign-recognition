import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# GTSRB has two official releases. torchvision.datasets.GTSRB downloads
# the smaller IJCNN 2011 competition version (26,640 train / 12,569 test).
# This project uses the larger, later "official" release instead:
# 39,209 train / 12,630 test. Confirmed via the official benchmark site
# (benchmark.ini.rub.de) and the dataset's actual host (sid.erda.dk) --
# these are the same URLs torchvision itself uses internally for other
# parts of the dataset.
GTSRB_BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
NUM_CLASSES = 43
IMAGE_SIZE = 224

DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_IMAGES_DIR = os.path.join(RAW_DIR, "GTSRB", "Final_Training", "Images")
TEST_IMAGES_DIR = os.path.join(RAW_DIR, "GTSRB", "Final_Test", "Images")
TEST_GT_CSV = os.path.join(TEST_IMAGES_DIR, "GT-final_test.csv")
ORGANIZED_TEST_DIR = os.path.join(DATA_DIR, "test_organized")

VIT_MODEL_NAME = "google/vit-base-patch16-224"
RESNET_MODEL_NAME = "resnet50"

VIT_CHECKPOINT = os.path.join(OUTPUTS_DIR, "vit_finetuned.pt")
RESNET_CHECKPOINT = os.path.join(OUTPUTS_DIR, "resnet_finetuned.pt")

VIT_RESULTS = os.path.join(OUTPUTS_DIR, "vit_results.json")
RESNET_RESULTS = os.path.join(OUTPUTS_DIR, "resnet_results.json")

QUANTIZATION_RESULTS = os.path.join(OUTPUTS_DIR, "quantization_results.json")
ROBUSTNESS_RESULTS = os.path.join(OUTPUTS_DIR, "robustness_results.json")

# Focal loss gamma -- how strongly to down-weight predictions the model
# already gets right easily, forcing it to spend more effort on the
# harder, rarer classes. 2.0 is the standard default from the original
# focal loss paper (Lin et al., 2017).
FOCAL_LOSS_GAMMA = 2.0

# IMPORTANT: every script in this project explicitly targets CPU, on
# purpose. Two reasons: (1) torch.quantization's dynamic INT8
# quantization is a CPU-focused feature with limited/no MPS support,
# and (2) saving a model while it's on an MPS device embeds
# device-specific info in the checkpoint, which breaks loading on
# machines without MPS. Training on CPU here avoids repeating that problem.
DEVICE = "cpu"