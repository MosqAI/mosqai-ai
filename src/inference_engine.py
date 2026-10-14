import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_INPUT_SIZE = 640
LETTERBOX_FILL = 114  # same grey Ultralytics pads with during training


class MosqAInferenceEngine:
    def __init__(self, model_path, confidence_threshold=0.5, nms_threshold=0.4, class_ids=None):
        self.session = ort.InferenceSession(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # None counts every class; set it to the mosquito class ids once a
        # multi-class model is used.
        self.class_ids = None if class_ids is None else set(class_ids)

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        height, width = model_input.shape[2], model_input.shape[3]
        # Dynamic axes come back as strings or None.
        self.input_height = height if isinstance(height, int) else DEFAULT_INPUT_SIZE
        self.input_width = width if isinstance(width, int) else DEFAULT_INPUT_SIZE

    def letterbox_geometry(self, h, w):
        """Return (scale, new_w, new_h, left, top) for fitting an h x w image into the input."""
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = round(w * scale), round(h * scale)
        return scale, new_w, new_h, (self.input_width - new_w) // 2, (self.input_height - new_h) // 2

    def letterbox(self, image):
        """Resize keeping the aspect ratio and pad to the model input size."""
        _, new_w, new_h, left, top = self.letterbox_geometry(*image.shape[:2])
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), LETTERBOX_FILL, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def preprocess(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not decode image at location: {image_path}")

        rgb_image = cv2.cvtColor(self.letterbox(original_image), cv2.COLOR_BGR2RGB)
        chw_img = np.transpose(rgb_image.astype(np.float32) / 255.0, (2, 0, 1))
        return np.expand_dims(chw_img, axis=0), original_image

    def postprocess_boxes(self, raw_output):
        """Boxes (x, y, w, h) in model input pixels from a YOLOv8 output of shape (1, 4 + num_classes, num_boxes)."""
        predictions = np.squeeze(raw_output, axis=0).T
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= self.confidence_threshold
        if self.class_ids is not None:
            keep &= np.isin(class_ids, list(self.class_ids))
        if not keep.any():
            return []

        x_center, y_center, width, height = predictions[keep, :4].T
        boxes = np.stack([x_center - width / 2, y_center - height / 2, width, height], axis=1)

        kept_indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences[keep].tolist(), self.confidence_threshold, self.nms_threshold
        )
        return [tuple(boxes[i]) for i in np.array(kept_indices).flatten()]

    def postprocess(self, raw_output):
        """Count detections in a YOLOv8 output."""
        return len(self.postprocess_boxes(raw_output))

    def detect(self, image_path, background_path=None):
        """Boxes (x, y, w, h) in original image pixels. The model looks at the frame
        alone, so background_path is accepted only to match the rules detector."""
        img_tensor, original_image = self.preprocess(image_path)
        raw_outputs = self.session.run(None, {self.input_name: img_tensor})
        scale, _, _, left, top = self.letterbox_geometry(*original_image.shape[:2])
        return [
            (round((x - left) / scale), round((y - top) / scale), round(w / scale), round(h / scale))
            for x, y, w, h in self.postprocess_boxes(raw_outputs[0])
        ]

    def count_mosquitoes(self, image_path, background_path=None):
        return len(self.detect(image_path, background_path))
