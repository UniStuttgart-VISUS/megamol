FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    sudo build-essential cmake cmake-curses-gui git

RUN apt-get update && apt-get install -y \
    curl zip unzip tar xorg-dev libgl1-mesa-dev libglu1-mesa-dev

ARG USER_ID 
ARG GROUP_ID
ARG USERNAME

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${USERNAME}

WORKDIR /home/${USERNAME}/megamol

# CMD mkdir build || true && cd build && \ 
#     cmake -DCMAKE_CXX_COMPILE=/usr/bin/g++-11 -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
#           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DMEGAMOL_VCPKG_DOWNLOAD_CACHE=ON .. && \
#     make -j 16 && make -j 16 install

CMD /bin/bash