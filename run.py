
from detect import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AppTech Rodent Detection')
    parser.add_argument(
        '--video_path', type=str, default="@20000102013000_20000102015959_9359.mp4", help='video dir')
    parser.add_argument(
        '--ann_dir', type=str, default="ann/", help='bbox score threshold')
    parser.add_argument(
        '--device', type=str, default="cpu", help='model location')
    parser.add_argument(
        '--score', type=int, default=0.97, help='model location')
    args = parser.parse_args()

    #time = (start_time, end_time,email_sent_time)
    sever = Server(args)
    sever.run()
    #sever.main(cam_dir="rtsp:root:pass@192.168.1.90/axis-media/media.amp?",top=50,show=True)



