#include "pch.h"
#include "STomoVolMassInfo.h"

STomoVolMassInfo::STomoVolMassInfo()
{
  Init();
}

STomoVolMassInfo::~STomoVolMassInfo()
{
  Reset();
}

STomoVolMassInfo::STomoVolMassInfo(const STomoVolMassInfo& Source)
{
  Init();
  _Copy(Source);
}

void	STomoVolMassInfo::operator=(const STomoVolMassInfo& Source)
{
  Reset();
  _Copy(Source);
}

void	STomoVolMassInfo::_Copy(const STomoVolMassInfo& Source)
{
  memcpy(dData, Source.dData, sdData);
}

void	STomoVolMassInfo::Reset(void)
{
  Init();
}

void	STomoVolMassInfo::Init(void)
{
  memset(dData, 0x00, sdData);
}


void  STomoVolMassInfo::VolToMass(const S3DPrinterInfo&printer_info)
{
  Vo_clad = printer_info.surface_area * printer_info.wall_thickness;
  Vo_core = Vo - Vo_clad;
  Mo_clad = Vo_clad * printer_info.Fclad * printer_info.PLA_density;
  Mo_core = Vo_core * printer_info.Fcore * printer_info.PLA_density;
  Mo = Mo_core + Mo_clad;

#ifdef _EZAIR_THETAC_ZERO
  Vss = Vtc - Vo;
#endif

  if(printer_info.bUseExplicitSS)
  {
    if (Vss > g_fMARGIN)
    {
      FLOAT32 radius = pow(Vss / (4.f / 3.4 * 3.141592f), 0.3333f);
      Ass_clad = 4. * 3.141592 * radius * radius;//support structure 부피를 sphere 형태라고 가정하고, Ass_clad를 sphere의 표면적으로 근사화한다.

      Vss_clad = Ass_clad * printer_info.wall_thickness;
      Vss_core = Vss - Vss_clad;
      Mss_clad = Vss_clad * printer_info.Fclad * printer_info.PLA_density;
      Mss_core = Vss_core * printer_info.Fcore * printer_info.PLA_density;
      Mss = Mss_core + Mss_clad;
    }
  }
  else//implicit
  {
    Mss = Vss * printer_info.Fss * printer_info.Css * printer_info.PLA_density;//obsolete..
 }

  Mbed = Vbed * printer_info.wall_thickness * printer_info.PLA_density;
  Mss += Mbed;//파이썬에 Mo, Mss만 전달하기 때문에, Mbed값은 Mss에 넣어주자.
  Mtotal = Mo + Mss;
}