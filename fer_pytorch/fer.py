import json
import time
import subprocess
import os
import numpy as np
import pandas as pd
import whisper


import warnings
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fer_pytorch.model import FERModel
from fer_pytorch.pre_trained_models import get_pretrained_model
from fer_pytorch.train_test_dataset import FERDataset
from keybert import KeyBERT

warnings.simplefilter(action="always")

EMOTION_DICT = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear"}


class FER:
    """
    The FER inference class.

    Implemented for inference of the Facial Emotion Recognition model on different types of data
    (image, list of images, video files and e.t.c.)
    """

    def __init__(self, size: int = 224, device: int = 0) -> None:
        self.device_id = device
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model: Optional[FERModel] = None
        self.mtcnn = MTCNN(keep_all=True, select_largest=True, device=self.device)

    def get_pretrained_model(self, model_name: str) -> None:
        """The method initializes the FER model and uploads the pretrained weights from the github page of the project.

        Args:
            model_name (str): The name that stands for the weights to be downloaded from the internet. The name
            coincides with the name of the model for convenience.
        """

        self.model = get_pretrained_model(model_name=model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_user_weights(self, model_arch: str, path_to_weights: str) -> None:
        """The method initializes the FER model and uploads the user weights that are stored locally.

        Args:
            model_arch (str): Model architecture (timm.list_models() returns a complete list of available models in
                timm).
            path_to_weights (str): Path to the user weights to be loaded by the model.
        """

        self.model = FERModel(model_arch=model_arch, pretrained=False)
        self.model.load_weights(path_to_weights)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(
        self, frame: Optional[np.ndarray], show_top: bool = False, path_to_output: Optional[str] = None
    ) -> List[dict]:
        """The method makes the prediction of the FER model on a single image.

        Args:
            frame (np.array): Input image in np.array format after it is read by OpenCV.
            show_top (bool): Whether to output only one emotion with maximum probability or all emotions with
            corresponding probabilities.
            path_to_output (str, optional): If the output path is given, the image with bounding box and top emotion
            with corresponding probability is saved.

        Returns:
            The list of dictionaries with bounding box coordinates and recognized emotion probabilities for all the
            people detected on the image.
        """

        fer_transforms = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet values
            ]
        )

        if frame is None:
            raise TypeError("A frame is None! Please, check the path to the image, when you read it with OpenCV.")

        result_list = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Calculating time taken to run face detection model
        t1=time.time()

        boxes, _ = self.mtcnn.detect(frame, landmarks=False)

        t2=time.time()
        face_det = t2-t1 #time for face detection

        #print("****************Time taken to run FD is :***************" , (t2-t1))

        #calculating time for Emotion detection
        t3= time.time()
        prob_limit = 0.95 #0.8,0.9
        if boxes is not None:
            for (x, y, w, h) in boxes:
                if not all(coordinate >= 0 for coordinate in (x, y, w, h)):
                    warnings.warn("Invalid face crop!")
                    continue

                image = gray[int(y) : int(h), int(x) : int(w)]
                image = Image.fromarray(image)
                image = fer_transforms(image).float()
                image_tensor = image.unsqueeze_(0)
                input = image_tensor.to(self.device)

                if self.model is not None:
                    with torch.no_grad():
                        output = self.model(input)
                else:
                    raise TypeError("Nonetype is not callable! Please, initialize the model and upload the weights.")
                probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()

                if show_top:
                    #Modification to add only certain frames to the result based on condition.
                    if (np.amax(probs[0])>prob_limit and EMOTION_DICT[probs[0].argmax()] != 'neutral'):
                    
                        result_list.append(
                            {
                                "box": [x, y, w, h],
                                "top_emotion": {EMOTION_DICT[probs[0].argmax()]: np.amax(probs[0])},
                            }
                        )

                    else:
                        pass
                else:
                    result_list.append(
                        {
                            "box": [x, y, w, h],
                            "emotions": {
                                EMOTION_DICT[0]: probs[0, 0],
                                EMOTION_DICT[1]: probs[0, 1],
                                EMOTION_DICT[2]: probs[0, 2],
                                EMOTION_DICT[3]: probs[0, 3],
                                EMOTION_DICT[4]: probs[0, 4],
                                EMOTION_DICT[5]: probs[0, 5],
                                EMOTION_DICT[6]: probs[0, 6],
                            },
                        }
                    )

                #Modification to get selected visualization.
                if (np.amax(probs[0])>prob_limit and EMOTION_DICT[probs[0].argmax()] != 'neutral'):
                    self.visualize(frame, [x, y, w, h], EMOTION_DICT[probs[0].argmax()], np.amax(probs[0]))
                else:
                    pass
        else:
            warnings.warn("No faces detected!")
        if path_to_output is not None:
            cv2.imwrite(path_to_output, frame)

        t4= time.time()
        emo_det = t4-t3 #time for face detection

        return result_list, face_det, emo_det
        
       

    def predict_list_images(
        self, path_to_input: str, path_to_output: str, save_images: bool = False
    ) -> List[Dict[str, Union[str, List[float], float]]]:
        """The method makes the prediction of the FER model on a list of images.

        Args:
            path_to_input (str): Path to the folder with images.
            path_to_output (str): Path to the output folder, where the json with recognition results and optionally
            the output images are saved.
            save_images (bool): Whether to save output images or not.

        Returns:
            The list of dictionaries with bounding box coordinates, recognized top emotions and corresponding
            probabilities for each image in the folder.
        """

        if not os.path.exists(path_to_input):
            raise FileNotFoundError("Please, check the path to the input directory.")

        os.makedirs(path_to_output, exist_ok=True)

        result_list = []
        path_to_output_file = None

        list_files = os.listdir(path_to_input)

        if len(list_files) == 0:
            warnings.warn(f"The input folder {path_to_input} is empty!")

        for file_name in tqdm(list_files):
            print(file_name)
            result_dict = {"image_name": file_name}

            if save_images:
                path_to_output_file = os.path.join(path_to_output, file_name)

            file_path = os.path.join(path_to_input, file_name)
            frame = cv2.imread(file_path)

            output_list = self.predict_image(frame, show_top=True, path_to_output=path_to_output_file)

            result_dict = self.preprocess_output_list(output_list, result_dict)
            result_list.append(result_dict)

        result_json = json.dumps(result_list, allow_nan=True, indent=4)

        path_to_json = os.path.join(path_to_output, "result.json")

        with open(path_to_json, "w") as f:
            f.write(result_json)

        return result_list
    
    


    #Changing fps to 30 from 25.
    def analyze_video(self, path_to_video: str, path_to_output: str, save_video: bool = False, fps: int = 30, frame_skip: int = 6) -> None:
        """The method makes the prediction of the FER model on a video file.

        The method saves the output json file with emotion recognition results for each frame of the input video. The
        json can be read further by Pandas and analyzed if it is needed.

        Args:
            path_to_video (str): Path to the input video file.
            path_to_output (str): Path to the output folder, where the json with recognition results and optionally
            the output video is saved.
            save_video (bool): Whether to save output video or not.
            fps (int): Number of fps for output video.
            frame_skip (int): The number of frames that user wants to skip. Example n=3,4,5,6 etc. Default value = 6.
            downsample_video (frame_widthxframe_height): Remember to maintain the aspect ratio closely. 
        """

        if not os.path.exists(path_to_video):
            raise FileNotFoundError("Please, check the path to the input video file.")

        result_list = []
        frame_array = []
        size = None

        filename = os.path.basename(path_to_video)

        print(f"Processing videofile {filename}...")

        os.makedirs(path_to_output, exist_ok=True)
       
        
        
        v_cap = cv2.VideoCapture(path_to_video) #path_to_video
        fps = (v_cap.get(cv2.CAP_PROP_FPS))
        
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(v_len)
        #Calculating time taken to run face and emotion detection model
        time_FD = []
        time_ED = []
        #frame_counter=0 #counter to skip frames
        #frame_skip = 6 #1,3,6,9 this is the frame skipping frequency
        for i in tqdm(range(v_len)):
            success, frame = v_cap.read()
            if not success:
                warnings.warn(f"The {i}-th frame could not be loaded. Continue processing...")
                continue

            height, width, layers = frame.shape
            size = (width, height)
            
            frame_array.append(frame)
            
            #skipping frames, tune the skip frequency, frame_id:i
            if i % frame_skip == 0:
                output_list, face_det, emo_det = self.predict_image(frame, show_top=True)
            
                time_FD.append(face_det)
                time_ED.append(emo_det)

                #Modification to get selected frames.
                #frame_array.append(frame)
                if len(output_list)!=0:

                    result_dict = {"frame_id": f"{i}", "time_stamp": round(i/30,1)}#rounding off to 1 decimal place.
                    #result_dict = {"time_stamp": i//30}
                    result_dict = self.preprocess_output_list(output_list, result_dict)
                    result_list.append(result_dict)

               

            

        print("Time taken for FD", np.sum(time_FD))
        print("Time taken for ED", np.sum(time_ED))

        result_json = json.dumps(result_list, allow_nan=True, indent=4)
        path_to_json = os.path.join(path_to_output, "result_ts.json")#Changed the output json filename

        with open(path_to_json, "w") as f:
            f.write(result_json)

        if save_video:
            path_to_video = os.path.join(path_to_output, filename) #path_to_video
            out = cv2.VideoWriter(path_to_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)#Changed codecc DIVX, path_to_video 
            print("Writing the output videofile...")
            for i in tqdm(range(len(frame_array))):
                out.write(frame_array[i])
            out.release()

        return time_FD, time_ED #to return the list in the output
    
    #Writing and combining new functions that includes :speech to text and matching time stamps from fer output video and speech to text
    
    def downsample_analyse(self, path_to_original_video: str, path_to_output: str, frame_width: int, frame_height: int, frame_skip_number: int, save_final_video: bool):
        #240X426, 360x640.
        os.makedirs(path_to_output, exist_ok=True)
        filename = os.path.basename(path_to_original_video)
        vidcap = cv2.VideoCapture(path_to_original_video)#filename
        #fps = (vidcap.get(cv2.CAP_PROP_FPS))
        f_h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        f_w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if f_h>f_w:
            f_width, f_height = frame_width, frame_height #videos like shorts.
        else:
            f_width, f_height = frame_height, frame_width #to make it size compatible. In arguments always give frame_width<frame_height.
        #os.makedirs(path_to_output, exist_ok=True)

        vid = os.system(f'ffmpeg -y -i {path_to_original_video} -vf scale={f_width}x{f_height}:flags=lanczos -c:v libx264 -preset slow -crf 21 -c:a copy {path_to_output}/downsample_{filename}')
        
        self.analyze_video(path_to_video=f'{path_to_output}/downsample_{filename}', path_to_output=f'{path_to_output}/final_'+os.path.splitext(str(filename))[0], save_video=save_final_video, frame_skip=frame_skip_number)        

    def time_stamps_video(self, result_file):
        #time_stamp_set = set()
        time_stamp_list = []
        emotion_list = []
        dict_time_emot = {}
        
        f = open(result_file)
        data = json.load(f)
        for i in range(len(data)):
            time_stamp_list.append(data[i]['time_stamp'])
            emotion_list.append(data[i]['emotion'])
        dict_time_emot=dict(zip(time_stamp_list, emotion_list))

        return dict_time_emot,time_stamp_list, emotion_list
    
    def speech_to_text(self, path_to_original_video):
        model = whisper.load_model("base")
        file = path_to_original_video
        result = model.transcribe(file, language="en", without_timestamps =False)
        print(str(file)+" is converted.")

        text = result['text']
        video_name = []
        sent_id = []
        time_start =[]
        time_end =[]
        sent =[]

        for i in range(len(result["segments"])):
            mydictionary = result["segments"][i]
            video_name.append(file)
            sent_id.append(i)
            time_start.append(mydictionary['start'])
            time_end.append(mydictionary['end'])
            sent.append(mydictionary['text'])
        #time_round = [round(i) for i in time_start]
        start_round = [round(i,1) for i in time_start]
        end_round = [round(i,1) for i in time_end]
        time_ranges = list(zip(start_round, end_round))

        return sent, time_ranges, sent_id


    def match_time(self, path_to_original_video, time_stamp_list, emotion_list, time_ranges, sent, dict_time_emot, path_to_output):
        filename = os.path.basename(path_to_original_video)
        final_sent_list = []
        final_time_stamp_list = []
        final_emotion_list =[]
        #There are chances of having emotion but no phrase.
        for key, values in dict_time_emot.items():
            for ind,ele in enumerate(time_ranges):
                if ele[0]<key<=ele[1]:
                    final_sent_list.append(sent[ind])
                    final_time_stamp_list.append(key)
                    final_emotion_list.append(values)
        kw_model = KeyBERT(model='all-mpnet-base-v2')
        key_phrases =[]
        for sentence in final_sent_list:
            keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words='english',
                                                highlight=False,top_n=3)
            keywords_list= list(dict(keywords).keys())
            key_phrases.append(keywords_list)

        df = pd.DataFrame({"time_stamp":final_time_stamp_list, "emotion": final_emotion_list, "text":final_sent_list, "key phrases":key_phrases})
        df.to_csv(f'{path_to_output}/emotion_' + os.path.splitext(str(filename))[0] + '.csv', index = False)
        return df
    
   

    #Version with downsampling

    def final_video_analyse(self, path_to_original_video: str, path_to_output: str, frame_skip_number: int, frame_width: int, frame_height: int, save_final_video: bool = False):
        filename = os.path.basename(path_to_original_video)
        print("Downsampling of video started...")
        self.downsample_analyse(path_to_original_video, path_to_output, frame_width, frame_height, frame_skip_number, save_final_video)

  
        dict_time_emot, time_stamp_list, emotion_list = self.time_stamps_video(f'{path_to_output}/final_'+os.path.splitext(str(filename))[0] + '/result_ts.json')
        print("Timestamps and emotions extracted...")

        print("Speech to text analysis started...")
        
        sent, time_ranges, sent_id = self.speech_to_text(path_to_original_video)
        print("Converted video speech to text...")
        
        print("Final emotion file preparation started...")
        self.match_time(path_to_original_video, time_stamp_list, emotion_list, time_ranges, sent, dict_time_emot, path_to_output)

        print("Emotion and speech analysis completed...")
    

    def run_webcam(self) -> None:
        """The method makes the prediction of the FER model for the stream from the web camera and shows the results in
        real-time."""

        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()

            if not success:
                warnings.warn("The frame could not be loaded. Continue processing...")
                continue

            output_list = self.predict_image(frame, show_top=True)
            print(output_list)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def test_fer(
        self,
        path_to_dataset: str = "fer_pytorch/dataset",
        path_to_csv: str = "fer_pytorch/dataset/new_test.csv",
        batch_size: int = 32,
        num_workers: int = 8,
    ) -> Dict[str, Any]:
        """The method is intended for convenient calculation of metrics (accuracy and f1 score) on the test part of the
        FER dataset.

        Args:
            path_to_dataset (str): Path to the folder with FER+ dataset.
            path_to_csv (str): Path to the csv file with labels for the test part.
            batch_size (int): Batch size for inference on the test part.
            num_workers (int): Number of workers to use while inference.

        Returns:
            The dictionary with accuracy and f1 score values.
        """

        test_fold = pd.read_csv(path_to_csv)

        test_transforms = A.Compose(
            [
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        test_dataset = FERDataset(test_fold, path_to_dataset=path_to_dataset, mode="test", transform=test_transforms)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pred_probs = []

        for _, (images, _) in enumerate(tqdm(test_loader)):
            images = images.to(self.device)
            if self.model is not None:
                with torch.no_grad():
                    y_preds = self.model(images)
            else:
                raise TypeError("Nonetype is not callable!")
            pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
        predictions = np.concatenate(pred_probs)

        test_fold["max_prob"] = predictions.max(axis=1)
        test_fold["predictions"] = predictions.argmax(1)

        accuracy = accuracy_score(test_fold["predictions"], test_fold["label"])
        f1 = f1_score(test_fold["predictions"], test_fold["label"], average="weighted")

        return {
            "accuracy": np.round(accuracy, 2),
            "f1": np.round(f1, 2),
        }

    @staticmethod
    def json_to_pandas(json_file: str) -> pd.DataFrame:
        """The helper method to transform output json file to Pandas dataframe in convenient way.

        Args:
            json_file (str): Path to json file.

        Returns:
            The Pandas dataframe.
        """
        return pd.read_json(json_file, orient="records")

    @staticmethod
    def preprocess_output_list(output_list: list, result_dict: dict) -> dict:
        """The method is intended to process output list with recognition results to make it more convenient to save
        them in json format.

        Args:
            output_list (list): Output list with result from the prediction on a single image.
            result_dict (dict): The dictionary that is modified as a result of this method and contains all the needed
            information about FER on a single image.

        Returns:
            The dictionary with FER results for a single image.
        """
        if output_list:
            output_dict = output_list[0]
            result_dict["box"] = [round(float(n), 2) for n in output_dict["box"]]
            result_dict["emotion"] = next(iter(output_dict["top_emotion"]))
            result_dict["probability"] = round(float(next(iter(output_dict["top_emotion"].values()))), 2)
        else:
            result_dict["box"] = []
            result_dict["emotion"] = ""
            result_dict["probability"] = np.nan
        return result_dict

    @staticmethod
    def visualize(frame: Optional[np.ndarray], box_coordinates: List[float], emotion: str, prob: float) -> None:
        """The function for easy visualization.

        Args:
            frame (Optional[np.ndarray]): Input frame.
            box_coordinates (list): The list with face box coordinates.
            emotion (str): Emotion output class from the fer model.
            prob (float): Emotion output probability from the fer model.
        """
        x, y, w, h = (
            box_coordinates[0],
            box_coordinates[1],
            box_coordinates[2],
            box_coordinates[3],
        )
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{emotion}: {prob:.2f}",
            (int(x), int(y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,#1.0,
            (0, 0, 255.0),
            2,#2,
        )
