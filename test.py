from torchaudio.io import StreamReader

# Path to the MP4 file
mp4_file_path = "./junk/-1eKufUP5XQ_4.mp4"


streamer = StreamReader(src=mp4_file_path)
streamer.add_basic_video_stream(
    frames_per_chunk=16000
    , frame_rate=25
    , width=512
    , height=512
    , format="rgb24"
)
n_ite = 3
video_chunk_list = []
for i,video_chunk in enumerate(streamer.stream()):
    if video_chunk is not None:
        video_chunk_list.append(video_chunk[0].float().numpy())
        if i + 1 == n_ite:
            break



print("video_chunk_list:",video_chunk_list)
