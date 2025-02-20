# f1tenth-racer
Code for f1tenth racing in ready-noticed path.
Participated in 2025 PNU-HYUNSONG F1Tenth Championship and won 1st prize. (team 적토마 from Pusan National University)

[![image](http://img.youtube.com/vi/pjeWsDWoA-A/maxresdefault.jpg)](https://www.youtube.com/watch?v=pjeWsDWoA-A)

### This project is based on the open-source project f1tenth-racing-stack-ICRA22, originally developed by zzjun725 for f1tenth racer and licensed under the MIT License. Modifications have been made to enhance functionality and performance.
https://github.com/zzjun725/f1tenth-racing-stack-ICRA22

# 1. Timetrial Node
This code gets single path (or 'lane' in this code) and follows it by stanley controller.
Result parameters are collected and will be printed when each lap is finished.

# 2. Hybrid Obstacle Node
This code gets multiple pathes to avoid obstacles. It will search for obstacles and find lanes that are free for obstacles found.
Once it determines which lane to follow, it will select appropriate controller for each situations that are pre-defined in the code.
Driving parameters for every lane is calculated each time and controller will be selected between 'Pure Pursuit', 'Stanley', 'LQR'.
