#!/usr/bin/env python
# coding: utf-8

# # Step 2: Create Job Submission Script
# 
# The next step is to create our job submission script. In the cell below, you will need to complete the job submission script and run the cell to generate the file using the magic `%%writefile` command. Your main task is to complete the following items of the script:
# 
# * Create a variable `MODEL` and assign it the value of the first argument passed to the job submission script.
# * Create a variable `DEVICE` and assign it the value of the second argument passed to the job submission script.
# * Create a variable `VIDEO` and assign it the value of the third argument passed to the job submission script.
# * Create a variable `PEOPLE` and assign it the value of the sixth argument passed to the job submission script.

# In[1]:


get_ipython().run_cell_magic('writefile', 'queue_job.sh', '#!/bin/bash\n\nexec 1>/output/stdout.log 2>/output/stderr.log\n\n# TODO: Create MODEL variable\nMODEL= $1\n# TODO: Create DEVICE variable\nDEVICE=$2\n# TODO: Create VIDEO variable\nVIDEO=$3\nQUEUE=$4\nOUTPUT=$5\n# TODO: Create PEOPLE variable\n\nmkdir -p $5\n\nif echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA\n    #Environment variables and compilation for edge compute nodes with FPGAs\n    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2\n\n    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh\n    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx\n\n    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3\nfi\n\npython3 person_detect.py  --model ${MODEL} \\\n                          --device ${DEVICE} \\\n                          --video ${VIDEO} \\\n                          --queue_param ${QUEUE} \\\n                          --output_path ${OUTPUT}\\\n                          --max_people ${PEOPLE} \\\n\ncd /output\n\ntar zcvf output.tgz *')


# # Next Step
# 
# Now that you've run the above cell and created your job submission script, you will work through each scenarios notebook in the next three workspaces. In each of these notebooks, you will submit jobs to Intel's DevCloud to load and run inference on each type of hardware and then review the results.
# 
# **Note**: As a reminder, if you need to make any changes to the job submission script, you can come back to this workspace to edit and run the above cell to overwrite the file with your changes.
