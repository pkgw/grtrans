  module fluid_model_hotspot_general
! Hotspot model from Schnittman & Bertschinger (2004)

  use class_four_vector
  use kerr, only: calc_rms, ledd, kerr_metric
  use phys_constants, only: pi

  implicit none

  namelist /hotspot/ rspot, r0spot, n0spot
  
  real :: rspot,r0spot,n0spot,tspot

  interface general_hotspot_vals
    module procedure general_hotspot_vals
  end interface

  interface init_general_hotspot
     module procedure init_general_hotspot
  end interface

  interface advance_general_hotspot_timestep
     module procedure advance_general_hotspot_timestep
  end interface

  contains

    subroutine read_general_hotspot_inputs(ifile)
    character(len=20), intent(in) :: ifile
    open(unit=8,file=ifile,form='formatted',status='old')
    read(8,nml=hotspot)
    close(unit=8)
!    write(6,*) 'hotspot: ',r0spot,rspot,n0spot
    end subroutine read_general_hotspot_inputs

    subroutine init_general_hotspot(ifile)
    character(len=20), intent(in), optional :: ifile
    character(len=20) :: default_ifile='hotspot.in'
    if (present(ifile)) then
       call read_general_hotspot_inputs(ifile)
    else
       call read_general_hotspot_inputs(default_ifile)
    endif
    tspot=0d0
    end subroutine init_general_hotspot

    subroutine advance_general_hotspot_timestep(dt)
    real, intent(in) :: dt
    tspot=tspot+dt
    end subroutine advance_general_hotspot_timestep

    subroutine general_hotspot_vals(x0,a,n)
      type (four_vector), intent(in), dimension(:) :: x0
      real, intent(in) :: a
      real, intent(out), dimension(size(x0)) :: n
      real, dimension(size(x0)) :: x,y,z,d2,t,phispot,xspot, & 
           yspot,r,phi,th,zero,zspot,thspot
      real :: dcrit,omega
!      write(6,*) 'hotspot sizes: ',size(x0), size(u)
      zero=0d0
      r = x0%data(2)
      phi = x0%data(4)
      t = x0%data(1)
      th = x0%data(3)
!      omega = 1d0/(r0spot**(3d0/2d0)+a)
! shift phi & t: (THIS SHOULD NOW BE DONE ELSEWHERE)
!      phi = -phi-pi/2d0
!      t = -t
! "cartesian" coords:
      x = r*sin(th)*cos(phi)
      y = r*sin(th)*sin(phi)
      z = r*cos(th)
      phispot = omega*(t+tspot)
      xspot = r0spot*sin(thspot)*cos(phispot)
      yspot = r0spot*sin(thspot)*sin(phispot)
      zspot=r0spot*cos(thspot)
! "distance" between geodesics & spot:
      d2 = (x-xspot)**2.+(y-yspot)**2.+(z-zspot)**2.
      dcrit = 16d0*rspot**2.
      n = merge(exp(-d2/2./rspot/rspot),zero,d2.lt.dcrit)
!      write(6,*) 'general hotspot r: ',r
!      write(6,*) 'general hotspot t: ',t
!      write(6,*) 'general hotspot th: ',th
!      write(6,*) 'general hotspot phi: ',phi
!      write(6,*) 'general hotspot d2: ',d2
!      write(6,*) 'general hotspot n: ',n
     end subroutine general_hotspot_vals

  end module fluid_model_hotspot_general