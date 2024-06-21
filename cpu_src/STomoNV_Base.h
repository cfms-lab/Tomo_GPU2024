#pragma once
#include "STPSlot.h"
#include "STomoVoxelSpaceInfo.h"
#include "S3DPrinterInfo.h"
#include "STomoVoxel.h"

using namespace Tomo;

class DLLEXPORT STomoNV_Base
{
public:
  STomoNV_Base();
  STomoNV_Base(const STomoNV_Base& Source);
  void	operator=(const STomoNV_Base& Source);
  void	_Copy(const STomoNV_Base& Source);
  ~STomoNV_Base();

  void	Reset(void);
  void	Init(void);

  S3DPrinterInfo    printer_info;
  STomoVolMassInfo  vm_info;
  TPSlotVector slotVec;
  STomoAABB2D AABB2D;

  void  Rotate(void); //rotate by (yaw, pitch, roll) and move to origin

  virtual void  Pixelize(const TVVector& CVV_vxls) {}
  virtual TPVector  slotsToPxls(enumPixelType _type) { TPVector tmp; return tmp;}//for rendering. time consuming. 
  virtual void      pxlsToSlots(TPVector& tri_pxls) {}
  virtual void  Pairing(void) {}//slot paring.
  virtual void  GenerateBed(void) {}
  virtual void  Calculate(void) {}//get Vss value
  //void  volToMass(void);

  virtual TPVector GetSSPixels(bool) { TPVector tmp; return tmp; }


  //closeVolumeVoxel
  TVVector CV_vxls;
  bool bStoreCVV;  



};
