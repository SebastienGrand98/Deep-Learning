import torch
import numpy as np
import cv2


class ObjectDetection:
    
    def __init__(self, video_path, result):
       
        self.video = video_path
        self.model = self.getModel()
        self.classes = self.model.names
        self.result = result
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)


    def getModel(self):

        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        return model


    def frameScore(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coord


    def getClass(self, x):

        return self.classes[int(x)]


    def plotBoxes(self, results, frame):

        labels, coord = results
        nb_labels = len(labels)

        frame_width, frame_height = frame.shape[1], frame.shape[0]
        for i in range(nb_labels):

            detection = coord[i]
            if detection[4] >= 0.2:
                x0, y0, x1, y1 = int(detection[0]*frame_width), int(detection[1]*frame_height), int(detection[2]*frame_width), int(detection[3]*frame_height)

                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, self.getClass(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame


    def __call__(self):

        cap = cv2.VideoCapture(self.video)
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        output = cv2.VideoWriter(self.result, four_cc, 20, (cap_width, cap_height))

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            results = self.frameScore(frame)
            frame = self.plotBoxes(results, frame)

            cv2.imshow("frame", frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break

            output.write(frame)



# Create a new object and execute.
path = f"/YOLO/video"
detection = ObjectDetection(path + "/test_video.mp4", path + "/output.avi")
detection()