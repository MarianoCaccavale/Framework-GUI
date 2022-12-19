import sys
import traceback
from tqdm import tqdm


def inference(path, conf_threshold=.50, IoU_threshold=.50):
    sys.path.append('./Models/yolov6')

    from yolov6.core.inferer import Inferer
    from yolov6.utils.nms import non_max_suppression

    # Set the corerct weight's name that the net will load
    weight_name = 'yolov6s.pt'

    # Path where store the weights
    weight_path = f"./Weights/yolov6_w/{weight_name}"

    inferer = Inferer(path, weight_path, 'cpu', "Models/yolov6/data/coco.yaml", 640, False)
    # Extract the setupped model from the Inferer
    model = inferer.model
    inferer.img_size = inferer.check_img_size(inferer.img_size, s=inferer.stride)  # check image size

    try:
        # For each file loaded(yep, I can load a directory and perform inference on each imgs)
        for img_src, img_path, vid_cap in tqdm(inferer.files):
            # Preprocess the images
            preproc_img, preproc_img_src = Inferer.precess_image(img_src, inferer.img_size, model.stride, False)

            # Batch it if it's only one img
            if len(preproc_img.shape) == 3:
                preproc_img = preproc_img[None]  # expand for batch dim

            # Make the predictions
            predictions = model(preproc_img)

            # Apply non-max-suppression
            detections = non_max_suppression(predictions, conf_threshold, IoU_threshold, [0])[0]

            # The model takes fixed size img, so before feeding them we have to preprocess them. Of course, the
            # resulting BB are in preprocessed coordinates, so we rescale them back
            detections[:, :4] = inferer.rescale(preproc_img.shape[2:], detections[:, :4], img_src.shape).round()

            return detections[..., :4], detections[..., 4:5], detections[..., 5:6]

    except:
        traceback.print_exc()

    return

