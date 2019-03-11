# ename lr tcontext num_epochs batchnorm split dropout model augment


#python main.py e15_f120_unet_dropout_1e-_TOAUGMENT 0.0001 120 20 no 0.8 yes unet yes

#python main.py e16_f120_unet_dropout_1e-4_TOAUGMENT_WITHBATCHNORM 0.0001 120 20 yes 0.8 yes unet yes

#python main.py e17_f120_unet_nodropout_1e-4_TOAUGMENT_RAW 0.0001 120 20 no 0.8 no unet yes

python main2.py e2_f120_onelinearlayer_allconvdropout 0.0001 120 20 yes 0.8 yes unet yes



