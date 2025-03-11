#!/bin/bash

docker build -t hzein86/quran-app .

docker push hzein86/quran-app:latest

docker context use darthfader

docker ps

docker stack deploy -c docker-compose-prod.yml quran-app

docker context use default

