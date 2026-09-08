# mp4 to emoji

This project extract mp4 clip into frames, calculate main character mask from gray background, then combine the mask file with original frame to change the background into transparent alpha channel data, so that emoji gif is generated.



# Demo

First you can get your video clip by any AI client such as DouBao(ByteDance)

![video_clip](emo_anjali_mudra.mp4)



Then you can run following command to generate gif from input mp4 file.

``` shell
mp4_redo.py --skip 3 emo_run_then_rest.mp4
```





![emo_anjali_mudra_and](emo_anjali_mudra_and.gif)

