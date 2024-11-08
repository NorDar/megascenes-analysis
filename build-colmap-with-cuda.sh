# https://colmap.github.io/install.html#installation
git clone https://github.com/colmap/colmap.git /tmp/colmap
cd /tmp/colmap
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install