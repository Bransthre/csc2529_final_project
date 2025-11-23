#!/bin/bash

USER_NAME=bransthr

watch -n 2 "
clear

echo -e '\n--- Number of Jobs Alive ---'
pending_job_count=\$(squeue -u ${USER_NAME} --states=PD | wc -l)
running_job_count=\$(squeue -u ${USER_NAME} --states=R | wc -l)
total_job_count=\$((pending_job_count + running_job_count - 2))


echo -e '\n--- Job Summary ---'
echo 'Pending jobs : ' \$((pending_job_count - 1))
echo 'Running jobs : ' \$((running_job_count - 1))
echo 'Total jobs   : ' \$total_job_count

echo -e '\n--- Current squeue ---'
squeue -u ${USER_NAME} -o '%.9i %.12P %.30j %.8u %.2t %.10M %.6D %R' --sort=-t
"