import os, cv2
import numpy as np

def read_anno_file(anno_file):
    assert os.path.exists(anno_file), "Annotation file does not exist!" + anno_file
    result = []
    with open(anno_file, 'r') as f:
        for line in f.readlines():
            items = {}
            items['vid'] = line.strip().split(',[')[0]
            labels = line.strip().split(',[')[1].split('],')[0]
            items['label'] = [int(val) for val in labels.split(',')]
            others = line.strip().split(',[')[1].split('],')[1].split(',')
            items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
            result.append(items)
    f.close()
    return result


def get_video_frames(video_file, topN=50):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
    print("original # frames: %d"%(len(video_data)))
    assert len(video_data) >= topN
    video_data = video_data[:topN]
    return video_data


if __name__ == "__main__":
    anno_file = "../videos/Crash-1500.txt"
    anno_data = read_anno_file(anno_file)

    video_path = "../videos/Crash-1500"
    for anno in anno_data:
        video_file = os.path.join(video_path, anno['vid'] + ".mp4")
        assert os.path.exists(video_file), "video file does not exist!" + video_file
        # read frames
        frames = get_video_frames(video_file, topN=50)
        labels = anno['label']
        print("file: %s, # frames: %d, # labels: %d."%(video_file, len(frames), len(labels)))
        print(len(labels))
        for idx, im in enumerate(frames):
           if labels[idx] == 1:
               cv2.putText(im, 'Accident', (int(im.shape[1] / 2)-60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
           else:
               cv2.putText(im, 'Normal', (int(im.shape[1] / 2)-60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
           cv2.imshow('frame', im)
           cv2.waitKey(100)