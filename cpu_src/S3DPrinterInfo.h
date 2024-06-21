#pragma once
#include "..\Tomo_types.h"
#include "STomoVoxel.h"
#include "STomoTriangle.h"

using namespace Tomo;

typedef DLLEXPORT struct
{
  FLOAT32 vtx0[3];//꼭지점의 상대좌표. (0,0)~(16,16) 및 노말벡터의 z 성분.
  FLOAT32 vtx1[3];
  FLOAT32 vtx2[3];
  FLOAT32 tri_nrm[3];
  FLOAT32 AABB[3];
  FLOAT32 MaxBoxSize;
} FlatTriInfo;

static const int nFlatTriInfoSize = sizeof(FlatTriInfo) / sizeof(FLOAT32);//match to 16 (BLOCK_SIZE)


class DLLEXPORT S3DPrinterInfo
{
public:
  S3DPrinterInfo(
    FLOAT32* _float32_x12 = nullptr,
    MESH_ELE_ID_TYPE* _int32_info = nullptr,
    MESH_ELE_ID_TYPE* _tri = nullptr, FLOAT32* _vtx = nullptr, FLOAT32* _vtx_nrm = nullptr, FLOAT32* _tri_nrm = nullptr,
    MESH_ELE_ID_TYPE* _chull_tri = nullptr, FLOAT32* _chull_vtx = nullptr, FLOAT32* _chull_trinrm = nullptr,
    FLOAT32 _yaw = 0, FLOAT32 _pitch = 0, FLOAT32 _roll = 0);
  S3DPrinterInfo(const S3DPrinterInfo& Source);
  void	operator=(const S3DPrinterInfo& Source);
  void	_Copy(const S3DPrinterInfo& Source);
  ~S3DPrinterInfo();

  static const int ndData = 16;
  static const int sdData = sizeof(FLOAT32) * ndData;
  union
  {
    struct {// Do not change this order!!
      FLOAT32	dVoxel, theta_c, surface_area, wall_thickness, \
        Fcore, Fclad, Fss, Css,
        PLA_density, BedInnerBound, BedOuterBound, BedThickness,
        //여기까지 12개는 파이썬에서 받아옴.
        yaw, pitch, roll;
    };
    FLOAT32 dData[ndData];
  };

  bool  bVerbose;//for debug
  bool  bUseExplicitSS;
  bool  bUseClosedVolumeVoxel;

  //input mesh data
  FLOAT32* rpVtx0, * pVtx1;//[nVtx]
  FLOAT32* rpTriNrm0, * rpVtxNrm0, * pNrm1, * pTriCenter;//[nTri]  triangle normal before/after rotation.
  MESH_ELE_ID_TYPE* rpTri0;//[nTri]

  MESH_ELE_ID_TYPE nVtx, nTri;//Cuastion: Do not treat this as int.
  int  nVoxel;//default 256

  size_t  nYPR;

  enumBedType BedType;

  //p-orbital (CvxHull)
  MESH_ELE_ID_TYPE nCHull_Tri, nCHull_Vtx;
  MESH_ELE_ID_TYPE* pCHull_Tri;
  FLOAT32* pCHull_Vtx, * pCHull_TriNrm, * pCHull_TriCenter;


#ifdef _USE_CUDA_FOR_TOMONV
  void  SetMaxTriDiameter(void);
  void  SetFlatTri(const TTriVector&);
  FLOAT32* pFlatTri;//CUDA pinned memory
  MESH_ELE_ID_TYPE nFlatTri;
  int  TriMaxDiameter;
  void  GetYPR4x3Matrix(FLOAT32* _YPR, int nCHullVtx, FLOAT32* _chull_vtx);
  void  GetYPR4x3Matrix(FLOAT32* _YPR, const TVVector& CVVoxels);
  FLOAT32* YPR_m4x3;
#endif

  void	Reset(void);
  void	Init(void);
  void  Set(FLOAT32* _float32_x12,
    MESH_ELE_ID_TYPE* _int32_info,
    MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm,
    MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trirnm,
    FLOAT32 _yaw, FLOAT32 _pitch, FLOAT32 _roll);

};