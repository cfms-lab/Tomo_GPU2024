# TomoNV example, C++ Dll version
from  tomoNV_Cpp import *
import os

#-----------------------------------------------
#(1) specify search conditon, filename and initial orientation. 
# VSCode에서 실행할 때는 .\\으로, VC++에서 디버깅할 때는 ..\\로 해야 함.
#========================================================================================================================
# DataSet= [ ('.\\Tomo_MeshData\\(1)cube50_nocut.obj', 0, 0, 0)]
# DataSet= [ ('.\\Tomo_MeshData\\(2)sphere75_3k.obj',   0, 0, 0)]
# DataSet= [ ('.\\Tomo_MeshData\\(3)cone50_63k.obj', 0,  0, 0)] 
DataSet= [ ('.\\Tomo_MeshData\\(4)Bunny_69k_2x.obj', 0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(5)Bunny_69k.stl',  0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(6)Bunny_69k_0.5x.stl',  0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(7)Bunny_5k.stl', 0, 0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(8)Bunny_1k.obj',   0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(9)manikin_rotated.obj',   0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(10)dragon_100k_1.5x.obj',   0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(11)happy_50k_0.75x.obj',   0,  0, 0)] 
# DataSet= [ ('.\\Tomo_MeshData\\(12)lucy_50k.obj',   0,  0, 0)] 

theta_YP = 0 #type 1). seeing a specific orientation. 
#theta_YP = 1  type 2). search optimal orientation, with search step 360/N (N=integer)
#========================================================================================================================
 
for Data in DataSet:
  (g_input_mesh_filename, Yaw, Pitch, Roll) = Data
  if( os.path.isfile(g_input_mesh_filename) ):
    if(theta_YP==0):# type 1). seeing a specific orientation.  
      nYPR_Intervals=1
      yaw_range   = np.ones(nYPR_Intervals) * toRadian(Yaw)
      pitch_range = np.ones(nYPR_Intervals) * toRadian(Pitch)
      roll_range  = np.ones(nYPR_Intervals) * toRadian(Roll)
    elif(theta_YP> 0): # type 2). search optimal orientation
      nYPR_Intervals= int(360 / theta_YP) +1
      yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
      pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
      roll_range  = np.zeros(1) 
    else:
      import sys
      sys.exit(0)

    tomoNV_Cpp1 = tomoNV_Cpp(g_input_mesh_filename, nYPR_Intervals, yaw_range, pitch_range, roll_range , bVerbose=True)# load mesh file and Cpp Dll
    #-----------------------------------------------
    # (2) specify 3D printer's g-code conditions.
    tomoNV_Cpp1.wall_thickness = 0.8 # [mm]
    tomoNV_Cpp1.PLA_density    = 0.00121 # density of PLA filament, [g/mm^3]
    tomoNV_Cpp1.Fclad   = 1.0 # fill ratio of cladding, always 1.0
    tomoNV_Cpp1.Fcore   = 0.15 # fill ratio of core, (0~1.0)
    tomoNV_Cpp1.Fss     = 0.2 # fill ratio of support structure, (0~1.0)
    tomoNV_Cpp1.Css     = 1.0 # correction constant for filament dilation effect. 
    tomoNV_Cpp1.dVoxel  = 1.0 # size of voxel. Do not change this value.
    tomoNV_Cpp1.nVoxel  = 256 # number of voxels. Do not change this value.
    tomoNV_Cpp1.theta_c = toRadian(60) #filament critical angle for support structure
    tomoNV_Cpp1.bUseExplicitSS = True #항상 True로 할 것.

    # tomoNV_Cpp1.BedType = ( enumBedType.ebtNone, 0, 0, 0) # 베드 없음
    # tomoNV_Cpp1.BedType = ( enumBedType.ebtSkirt, 3, 3+0.8, 0.2)# 스커트. 라인 간격, 스커트 라인 간격 + 스커트라인 수 *필라멘트 두께 , 바닥 레이어 높이
    # tomoNV_Cpp1.BedType = ( enumBedType.ebtBrim, 0, 10 * 0.4, 0.2)# 브림. 라인  수 * 서피스 레이어 두께  , 바닥 레이어 높이
    tomoNV_Cpp1.BedType = ( enumBedType.ebtRaft, 0, 2, 0.3 + 0.27 + 2 * 0.2)# 0, 래프트 크기(mm), 베이스 두께 + 인터페이스 두께  + 서피스 레이어 수 * 서피스 레이어 두께
    
    # tomoNV_Cpp1.Run(cpp_function_name = 'TomoNV_INT3') #call CPU version
    tomoNV_Cpp1.Run(cpp_function_name = 'TomoNV_CUDA') #call GPU version. Need NVIDIA graphic card (GTX760 or above)
    
    #-----------------------------------------------
    # (3) Rendering

    if tomoNV_Cpp1.bVerbose:
      Plot3DPixels(tomoNV_Cpp1) #show pixels of the 1st optimal
      # PrintSlotInfo( tomoNV_Cpp1, X=2,Y=2)
      #tomoNV_Cpp1.Print_table()
      #print( tomoNV_Cpp1.YPR)
      #print( tomoNV_Cpp1.Mss3D)

    if tomoNV_Cpp1.nYPR_Intervals >= 5:
      (optimal_YPRs,worst_YPRs) = findOptimals(tomoNV_Cpp1.YPR, tomoNV_Cpp1.Mtotal3D, g_nOptimalsToDisplay) # 결과값으로부터 최적 배향 찾기
      Plot3D(tomoNV_Cpp1.mesh0, yaw_range, pitch_range, tomoNV_Cpp1.Mtotal3D, optimal_YPRs, worst_YPRs) # 찾은 최적 배향 그래프 출력.
  else:
    print( 'File Does Not Exists!!' + g_input_mesh_filename)
#-----------------------------------------------
