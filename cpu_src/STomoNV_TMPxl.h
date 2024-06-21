#pragma once
#include "STPSlot.h"
#include "STomoVoxelSpaceInfo.h"
#include "STomoNV_Base.h"

using namespace Tomo;

class DLLEXPORT STomoNV_TMPxl : public STomoNV_Base
{
public:
  STomoNV_TMPxl();
  //STomoNV(S3DPrinterInfo _info);
  STomoNV_TMPxl(const STomoNV_TMPxl& Source);
    void	operator=(const STomoNV_TMPxl& Source);
    void	_Copy(const STomoNV_TMPxl& Source);
  ~STomoNV_TMPxl();

  void	Reset(void);
  void	Init(void);
  
  //void  Rotate(void);//(yaw, pitch, roll) 회전 후 origin으로 평행이동.
  void  Pixelize(const TVVector& CVV_vxls);//픽셀화 후 slot에 바로 넣기.
    void triPixel(
    FLOAT32* v0, FLOAT32* v1, FLOAT32* v2,
    FLOAT32* n0, FLOAT32* n1, FLOAT32* n2,
    TPVector& tri_pxls);
    TPVector  slotsToPxls(enumPixelType _type);//for rendering. time consuming. 
    void      pxlsToSlots(TPVector& tri_pxls);
  void  Pairing(void);//slot paring.
  void  Calculate(void);//get Vss value

  TPVector GetSSPixels(bool _bUseExplicitSS);
  void  GetCVVoxelsFromSlots(TVVector& CV_vxls);//make CV_vxls from slots

};
