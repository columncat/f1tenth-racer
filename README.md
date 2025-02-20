# f1tenth-racer
Code for f1tenth racing in ready-noticed path.
Participated in 2025 PNU-HYUNSONG F1Tenth Championship and won 1st prize. (team 적토마 from Pusan National University)  
이 레포지터리는 부산대학교에서 개최된 2025 PNU-현송 f1tenth 챔피언십에 출전한 부산대 적토마 팀의 코드입니다.

[![image](http://img.youtube.com/vi/pjeWsDWoA-A/maxresdefault.jpg)](https://www.youtube.com/watch?v=pjeWsDWoA-A)

### This project is based on the open-source project f1tenth-racing-stack-ICRA22, originally developed by zzjun725 for f1tenth racer and licensed under the MIT License. Modifications have been made to enhance functionality and performance.
https://github.com/zzjun725/f1tenth-racing-stack-ICRA22

# 1. Timetrial Node
This code gets single path (or 'lane' in this code) and follows it by stanley controller.  
Result parameters are collected and will be printed when each lap is finished.

# 2. Hybrid Obstacle Node
This code gets multiple pathes to avoid obstacles. It will search for obstacles and follow lanes that are free from obstacles.


### New feature 1
Once it determines which lane to follow, it will select appropriate controller for each situations that are pre-defined in the code.  
Driving parameters for every lane is calculated each time and controller will be selected between 'Pure Pursuit', 'Stanley', and 'LQR'.

For example, assume that three pathes 'optimal'(main lane), 'in_course', 'out_course' are being used.  
For every path, the code checks whether each lane is free of obstacle (is_free) and whether ego vehicle is on the lane (is_on_lane).
Then we can set controller selection algorithm arbitrarily like the example below.

1. If optimal lane is free and ego vehicle is on the optimal lane.  
: This case you can select LQR for optimal driving.

3. If optimal lane is free, but ego vehicle is NOT on the optimal lane.  
: This case you can select Stanley to gradually decrease cross-track error.

4. If optimal lane is NOT free while any other lane is free, and ego vehicle is on any lane.  
: This case you can select Stanley or LQR to keep follow the lane.

4. If optimal lane is NOT free while any other lane is free, but ego vehicle is NOT on any lane.  
: This case you can select Pure Pursuit to switch lane quickly.

5. If none of the lanes are free.  
: This case you may choose to stop or drive backwards.
In this case, you can use driving messeage queue to assure your vehicle to continue driving backwards for few iterations. (HybridNode.message_queue)


### New feature 2
This code calculates inner product between unit vector of ego vehicle's global yaw and unit vector of direction from ego vehicle to obstacle.  
Only obstacles with inner product value over 0.9(in this code or preferred cosine value for desired angles) will be detected as obstacles.



