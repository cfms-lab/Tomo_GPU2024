#include "step3_generateBed.cuh"
#include "atomicWrite.cuh"

using namespace Tomo;

__device__ __inline__ float cu_dist2D(int x0 , int y0 , int x1 , int y1)
{
	return sqrt( ((float(x0) - x1)*(float(x0) - x1) + (float(y0) - y1)*(float(y0) - y1)));
}

__global__ void cu_genBed(
	int nVoxelX, int nSlot, //constants
	enumBedType bedtype, float outerRadius, float innerRadius, float height, //bed parameter
	CU_SLOT_BUFFER_TYPE* cu_nPixl, //slot data
	CU_SLOT_BUFFER_TYPE* cu_pType, 
	CU_SLOT_BUFFER_TYPE* cu_pZcrd, 
	CU_SLOT_BUFFER_TYPE* cu_pZnrm, 
	CU_SLOT_BUFFER_TYPE* cu_Vo, 
	CU_SLOT_BUFFER_TYPE* cu_Vss)
{
	//Find nearby nonzero pxl to make bed structure. each thread takes care of 1 slot 
	const int slotID		= blockIdx.x * blockDim.x + threadIdx.x;
	if(slotID >= nSlot) return;
	
	const CU_ULInt nSlotData	= nSlot * CU_SLOT_CAPACITY_16;

	CU_ULInt uliSlotIdx				= slotID;
	CU_ULInt uliSlotDataIdx		= slotID * CU_SLOT_CAPACITY_16;

	CU_SLOT_BUFFER_TYPE* memNPxl	= cu_nPixl	+ uliSlotIdx;
	CU_SLOT_BUFFER_TYPE* memVo		= cu_Vo			+ uliSlotIdx;
	CU_SLOT_BUFFER_TYPE* memVss		= cu_Vss		+ uliSlotIdx;
	CU_SLOT_BUFFER_TYPE* memType	= cu_pType	+ uliSlotDataIdx;
	CU_SLOT_BUFFER_TYPE* memZcrd	= cu_pZcrd	+ uliSlotDataIdx;
	CU_SLOT_BUFFER_TYPE* memZnrm	= cu_pZnrm	+ uliSlotDataIdx;
	int S_L = *memNPxl;
	
	if(S_L > 0 )
	{
		int p0 = S_L-1;//last pxl, on bottom plate. 
		int p0_zcrd = memZcrd[p0];
		int p0_type = memType[p0];
		if(p0_zcrd == 0)
		{
			if( bedtype == enumBedType::ebtRaft && ((p0_type & cu_typeBe) || (p0_type & cu_typeSSA)) )
			{
				//����Ʈ�� ���� beta�Ʒ� �� �� �� ����ش�.
			}
			else if((p0_type & cu_typeAl)  || (p0_type & cu_typeBe)
					 || (p0_type & cu_typeSSA) || (p0_type & cu_typeSS)  )  
		  {
				return;//pass. this pxl cannot become bed. 
			}
		}
	}


	int Xcrd = slotID % nVoxelX;
	int Ycrd = slotID / nVoxelX;
	int I0 = max( Xcrd - int(outerRadius), 0);
	int I1 = min( Xcrd + int(outerRadius), nVoxelX);
	int J0 = max( Ycrd - int(outerRadius), 0);
	int J1 = min( Ycrd + int(outerRadius), nVoxelX);

	//compare distance to nearby target non-zero-type pxl
	float min_dist = 1e5;
	for( int i = I0 ; i < I1 ; i++)	
	{
		for( int j = J0 ; j < J1 ; j++)	
		{
			int slotID = j * nVoxelX + i;
			SLOT_BUFFER_TYPE S_L_tgt = *(cu_nPixl + slotID);
			if(S_L_tgt > 0)
			{
				int p = S_L_tgt-1;// last pxl, on bottom plate.
				CU_ULInt uliSlotDataIdx_tgt		= slotID * CU_SLOT_CAPACITY_16;
				CU_SLOT_BUFFER_TYPE* memType_tgt	= cu_pType	+ uliSlotDataIdx_tgt;
				CU_SLOT_BUFFER_TYPE* memZcrd_tgt	= cu_pZcrd	+ uliSlotDataIdx_tgt;

				CU_SLOT_BUFFER_TYPE p_type_tgt = memType_tgt[p];
				CU_SLOT_BUFFER_TYPE p_z_tgt = memZcrd_tgt[p];
				float dist = cu_dist2D( Xcrd, Ycrd, i, j);
				if( p_z_tgt == 0 && 
					((p_type_tgt & cu_typeAl) || (p_type_tgt & cu_typeBe) || (p_type_tgt & cu_typeSSA) || (p_type_tgt & cu_typeSS) ) )
				{
					min_dist = min( min_dist, dist);  
				}
			}//end of if(S_L_tgt > 0)
		}//end of for(j..)
	}//end of for(i..)

	if(min_dist > outerRadius) return;

	bool bBedPxl = false;
	if(     bedtype == enumBedType::ebtSkirt && min_dist > innerRadius) bBedPxl = true;
	else if(bedtype == enumBedType::ebtBrim)   bBedPxl = true;  
	else if(bedtype == enumBedType::ebtRaft)   bBedPxl = true;  

	if(bBedPxl && S_L < CU_SLOT_CAPACITY_16-1) 	
	{
		//insert new bed pxl
		cu_Add(		memNPxl, 1); S_L++;
		cu_Or(		memType + S_L-1, cu_typeBed);
		cu_Exch(	memZcrd + S_L-1, 0);

		//update volume
		cu_Add(		memVss, /*int(height)�� �ϸ� 0 �� ���ͼ� �ȵȴ�*/ 1);
	}

}