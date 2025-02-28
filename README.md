# Stereovision Algorithm Comparison

## Project Description
This project compares different stereovision algorithms for disparity map estimation and real-time distance measurement. The implemented methods include:  
- **StereoBM**  
- **StereoSGBM**  
- **Hybrid-based - Cascade classifier combined with average color analysis of a selected area**
- **FastACVNet**  

This project was developed as part of my master's thesis.
    
## Getting Started

1. Clone the repository
```shell
git clone https://github.com/sailor-elite/Stereovision_AlgorithmComparison.git
cd Stereovision_AlgorithmComparison
   ```
2. Install dependencies:
    
```shell
pip install -r requirements.txt
```
    
3. Run the main script:

```shell
python main.py
```
- In main.py, set the variables imgLeft and imgRight to the corresponding stereo image paths.
- Adjust parameters as needed.
- Adjust the IDs of your left and right cameras and run:
```shell
python Taking_Stereovision_Picture.py
```
- Adjust the IDs of your left and right cameras and run:
```shell
python Real_time_distance_measurement.py
```
## Results of the  hybrid based algorithm
<p align="center">
<img src="data/Result_realtime.png" />
</p>
<p align="center">
<img src="data/result_disp_map.png" />
</p>

