docker run -it --user=root \
  --runtime nvidia \
  --rm \
  --network host \
  --ipc=host \
  --gpus all \
  -v ./:/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/root/.Xauthority \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  ait4automation