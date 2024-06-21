#include "../cpu_src/STomoNV_Base.h" 
#include "CUDA_types.cuh"
#include "SlotData.cuh"

using namespace Tomo;

typedef int* INTP;
typedef float* FLOATP;

//This class uses only 1(Default) CUDA stream.

class  STomoNV_CUDA0 : public STomoNV_Base
{
public:
  //derived from STomoNV_Base class
  __host__ STomoNV_CUDA0();
  __host__ STomoNV_CUDA0(const STomoNV_CUDA0& Source);
  __host__ void	operator=(const STomoNV_CUDA0& Source);
  __host__ void	_Copy(const STomoNV_CUDA0& Source);
  __host__ ~STomoNV_CUDA0();

  __host__ void	Reset(void);
  __host__ void	Init(void);

    __host__ void	Init_CUDA(void);
      __host__  void  initCudaMem(void);
      __host__  void  clearCudaMem(void);
      __host__  void	getBatchNumber(void);

  __host__ void  Run(float *_Mo, float *_Mss);//run for nYPR
    __host__ void  RotateAndPixelize(int _nYPRInBatch,int yprID_to_start);
    __host__ void  Pairing(int _nYPRInBatch);
    __host__ void ReducedSum(int _nYPRInBatch, float *_Mo, float *_Mss);


  __host__ FLOAT32* ReadPxls(int _yprID, int* _nData2i, INT16** _pData2i);

  static const size_t niData = 20;
  static const size_t siData = sizeof(size_t) * niData;
  union
  {
    struct 
    { 
      //params
      size_t nFlatTri;
      size_t nVoxelX,nVoxelY;//= fixed at 256 x 256.
      size_t nSlot;//=nVoxelX * nVoxelY = 255 * 255 = 65,536
      size_t nYPR;
      size_t nSlotData;

      size_t nWorksInBatch;//actual number of YPR's to process at a time

      //data size
      size_t  s_nSlot;
      size_t  s_nSlotData;
      size_t  s_nFlatTri;

      size_t  s_m4x3;
      size_t  gpu_available_mem;
      size_t  gpu_total_mem;

      //device property
      size_t  nMultiProcessor;//==6 for GTX760
      size_t  nMaxThreadPerMP;//==1024 for GTX760

      size_t  nMaxThreads;// = nMultiProcessor * nMaxThreadPerMP = 12,288 for GTX760
      size_t  nMaxConcurrentStream;//max number of streams. == 16 for GTX760
      size_t  nStream;//actual number of streams to use concurrently. <= nMaxConcurrentStream 

    };
    size_t iData[niData];
  };

  //Slot data
  SlotData  cpu_SlotData;
  std::vector<SlotData> gpu_SlotDataVec;

  static const int cu_nfpData = 3;
  static const int cu_sfpData = sizeof(FLOATP) * cu_nfpData;
  union
  {
    struct
    {
      FLOATP cu_FlatTri0;//input triangle data from CPU [nTri * nFlatTriInfoSize]
#ifndef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP
      FLOATP cu_FlatTri1;//triangle data after rotation. [(nTri * nFlatTriInfoSize) * nYPR]. for debug.
#endif
      FLOATP  cu_m4x3;//rotation + trnslation matrix. [nYPR * 4*4] 
    };
    FLOATP cu_fpData[cu_nfpData];
  };

  
};
