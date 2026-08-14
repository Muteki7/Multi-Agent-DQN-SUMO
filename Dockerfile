#Built on ubuntu 24.04 rahter than python, since a lot of sumo libraries are built in c++
#So pip install won't work for them we need to build them from source, Linux works natively well for this 
#since we have apt to install will build the dependicens more cleanly and will make sure nothing crashes 

FROM ubuntu:24.04
 
ENV DEBIAN_FRONTEND=noninteractive \
    SUMO_HOME=/usr/share/sumo \
    PYTHONUNBUFFERED=1

#Need to download sumo tools
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git \
    && apt-get install -y sumo sumo-tools \
    && rm -rf /var/lib/apt/lists/*

#Building workdir first and then switch over user access
WORKDIR /app

COPY requirements.txt .

#Installs libraries globablly but since this will be run in a container the global just means the container so this is no big deal
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# sumo-rl's PyPI release (1.4.5) is currently broken against recent PettingZoo -- `from pettingzoo.utils import agent_selector` raises
# TypeError because PettingZoo renamed the class to AgentSelector. Fixed on
# GitHub main, not yet released to PyPI, so downloaded directly from the library

RUN pip install --no-cache-dir --break-system-packages \
        "git+https://github.com/LucasAlegre/sumo-rl.git"

COPY . .

#Create non-root user and give ownership of /app
RUN useradd --create-home app && chown -R app:app /app

USER app

#Run docker and tensorboard
EXPOSE 6006

CMD ["python3", 'train.py']
