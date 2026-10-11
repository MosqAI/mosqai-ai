import cv2
import numpy as np 
import onnxruntime as ort

class MosqAInferenceEngine:
    def __init__(self,model_path,confidence_threshold=0.5,nms_threshold=0.4):
        self.session = ort.InferenceSession(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.input_name = self.session.get_input()[0].name
        self.input_shape = self.session.get_input()[0].shape 

    def preprocess(self,image_path):
        original_image = cv2.imread(image_path)

        if original_image is None:
            raise FileNotFoundError(f"Could not Load Image at location: {image_path}")

        resized_img = cv2.resize(original_image,(640,640))
        rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        img_norm = rgb_image.astype(np.float32) / 255.0

        tensor_img = np.expand_dims(chw_img, axis=0) # The reason I did this was because Tensor expects [1, 3, 640, 640] for it to work with ONNX
        return tensor_img, original_img

    def count_mosquitoes(self,image_path):
        img_tensor,original_img = self.preprocess(image_path)

        if image_path or original_img is None:
            raise ValueError(f"Expected some values buit got none Please look into you image path")

        raw_outputs = self.session.run(None, {self.input_name: input_tensor})

        output_matrix = raw_outputs[0][0]

        boxes=[]
        confidences =[]

        for row in output_matrix:
            classes_row = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence >= self.conf_threshold:
                x_center, y_center, width, height = row[0:4]

                x_min = (x_center - (width / 2))
                y_min = (y_center - (height / 2))

                boxes.append([x_min,y_min,int(width),int(height)])
                confidences.append(float(confidence))

        suppressed_indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        mosquito_count = len(suppressed_indices)
        return mosquito_count