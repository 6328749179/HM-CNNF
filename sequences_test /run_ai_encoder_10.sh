echo 'Processing   '$1' QP'$2
../bin/TAppEncoder -c ../cfg/encoder_intra_main10_cnnf.cfg -c ../cfg/per-sequence/$1.cfg -q $2 -b ../encoder_out/bin/enc_$1_AI_Q$2.bin -o ../encoder_out/recyuv/enc_$1_AI_Q$2.yuv 2>&1| tee ../encoder_out/log/enc_$1_AI_Q$2.log
