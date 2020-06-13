#%%writefile person_detect.py
#creates job submission file

%%writefile person_detect.py

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            return frame
    
    def check_coords(self, coords,initial_w,initial_h):
        d={k+1:0 for k in range(len(self.queues))}
        dummy = ['0', '1' , '2', '3']
        for coord in coords:
            xmin = int(coord[3] * initial_w)
            ymin = int(coord[4] * initial_h)
            xmax = int(coord[5] * initial_w)
            ymax = int(coord[6] * initial_h)
            dummy[0] = xmin
            dummy[1] = ymin
            dummy[2] = xmax
            dummy[3] = ymax
            for i, q in enumerate(self.queues):
                if dummy[0]>q[0] and dummy[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.input_name=None

        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.ie=IECore()
        self.network = self.ie.load_network(network=self.model, device_name=self.device, num_requests=1) 
        supported_layers_path = self.ie.query_network(network=self.model, device_name=self.device)
        keys=self.model.layers.keys()
        for l in keys:
            unsupported_layers_path=""
            if l not in supported_layers_path:
                unsupported_layers_path=l
        if len(unsupported_layers_path) != 0:
            sys.exit(1) 
       
                
        
    def predict(self, image,w,h):
        input_img=image
        image = self.preprocess_input(image)
              
        
        input_dict={self.input_name: image}  

        infer_request_handle = self.network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
        
        coords, image = self.draw_outputs(outputs, input_img,w,h)    
        return coords, input_img
    
    def draw_outputs(self, coords, image,w,h):
        det = []        
        for obj in coords[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            if obj[2] > self.threshold:
                xmin = int(obj[3] * w)
                ymin = int(obj[4] * h)
                xmax = int(obj[5] * w)
                ymax = int(obj[6] * h)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
                det.append(obj)
        return det, image

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        return image

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame,initial_w,initial_h)
            num_people= queue.check_coords(coords,initial_w, initial_h)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)
