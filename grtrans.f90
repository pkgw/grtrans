
       subroutine grtrans_main(ifile,outfile)
       use omp_lib
       use grtrans_inputs
       use grtrans, only: grtrans_driver
       use ray_trace
       use fluid_model, only: load_fluid_model, unload_fluid_model, &
            advance_fluid_timestep, source_params, assign_source_params_type, &
            fluid_args, assign_fluid_args
       use emissivity, only: emis_params
       use geodesics, only: initialize_pixels, geokerr_args, &
         del_geokerr_args,initialize_geokerr_args, initialize_geo_tabs
       use chandra_tab24, only: load_chandra_tab24, del_chandra_tab24

       implicit none
       character(len=40), intent(in) :: outfile,ifile
!       character(len=40) :: outfile,ifile
       integer :: nextra=0, inum, gunit, i, ncams, j, m, l, nparams
       type (ray_set), dimension(:), allocatable :: c
       type (geokerr_args) :: gargs
       type (fluid_args) :: fargs
       type (source_params), dimension(:), allocatable :: sparams
       type (emis_params) :: eparams
       character(len=20), dimension(3) :: knames,kdescs
       knames(1)='nx'; knames(2)='ny'; kdescs(1)='# x pixels'
       kdescs(2)='# y pixels'
       knames(3)='nu'; kdescs(3)='Frequency (Hz)'
       gunit=12
       call read_inputs(ifile)
       write(6,*) 'dfile: ',fdfile
       if(extra==1) nextra=13
! these can later be added to a loop over emis parameter structures
!       eparams%gmin=gmin; 
       eparams%gmax=gmax; eparams%p1=p1
       eparams%p2=p2; 
!       eparams%mu=muval
       nparams=size(mdots)
       allocate(sparams(nparams))
       do i=1,nparams
          sparams(i)%mdot=mdots(i); sparams(i)%mbh=mbh
          sparams(i)%nfac=2.; sparams(i)%bfac=70.
          sparams(i)%jetalphaval=jetalpha
          sparams(i)%gminval=gmin; sparams(i)%muval=muval
          sparams(i)%gmax=gmax
          sparams(i)%p1=p1
          sparams(i)%p2=p2
          call assign_source_params_type(sparams(i),stype)
       enddo
       NCAMS=nparams*nfreq*nmu*nt
       allocate(c(NCAMS))
       write(6,*) 'outfile grtrans: ',outfile, NCAMS
       do m=1,NCAMS
         call initialize_raytrace_camera(c(m),nro,nphi,nvals,nextra)
       enddo
 !      write(6,*) 'spin: ',spin
       call assign_fluid_args(fargs,fdfile,fhfile,fgfile,fsim,fnt,findf,fnfiles,fjonfix, &
            fnw,fnfreq_tab,fnr,foffset,fdindf,fmagcrit,frspot,fr0spot,fn0spot,ftscl,frscl, &
            fwmin,fwmax,ffmin,ffmax,frmax,fsigt,ffcol,fmdot,mbh)
       call load_fluid_model(fname,spin,fargs)
!       do l=1,1
!          do j=1,1
       if(nup.eq.1.and.nvals.eq.4) call load_chandra_tab24()
       do j=1,nmu
          call initialize_geokerr_args(gargs,nro*nphi)
          call initialize_pixels(gargs,use_geokerr,standard,mu0(j),phi0,spin,uout,uin, &
               rcut,nrotype,a1,a2,b1,b2,nro,nphi,nup)
          if(nload.gt.1) then
          ! loop over geodesics calculating first point to get t0 for everything
             gargs%nup=1
             write(6,*) 'grtrans nload gt 1 loop',size(gargs%t0),allocated(gargs%t0),gargs%nup
!$omp parallel do private(i) shared(gargs)
             do i=1,c(1)%nx*c(1)%ny
                call initialize_geo_tabs(gargs,i)
             enddo
!$omp end parallel do
             write(6,*) 'grtrans after init geo tabs',minval(gargs%t0),maxval(gargs%t0)
             where(gargs%t0.gt.0.)
                gargs%t0=gargs%t0-minval(gargs%t0)
             elsewhere
                gargs%t0=0.
             endwhere
             gargs%nup=nup
             write(6,*) 'grtrans after init geo tabs',minval(gargs%t0),maxval(gargs%t0)
          else
             gargs%t0=0.
          endif
          do l=1,nt
!       write(6,*) 'pre loop spin: ',spin,gargs%a
!$omp parallel do private(i) shared(gargs,gunit,c,j,nt,l,spin,iname,ename,fname,sparams,eparams,nfreq,nparams,freqs,nup)
             do i=1,c(1)%nx*c(1)%ny
!                write(6,*) 'i: ',i
!              do i=7128,7128
!                 write(6,*) 'i: ',i
!                write(6,*) 'after loop spin: ',mdots(1),mbh
                call grtrans_driver(gargs,gunit,c,i,(j-1)*nt+l,iname,ename,fname, &
                  sparams,eparams,nfreq,nparams,freqs,nup,extra)
             enddo
!       write(6,*) 'after loop i'
!$omp end parallel do
!       write(6,*) 'del geokerr args before'
             if(l.lt.nt) call advance_fluid_timestep(fname,dt)
          enddo
          spin=gargs%a
          call del_geokerr_args(gargs)
!       write(6,*) 'spin after: ',spin
!          call advance_fluid_timestep(fname,dt)
       enddo
       if(nup.eq.1.and.nvals.eq.4) call del_chandra_tab24()
!       write(6,*) 'Write camera'
       do m=1,ncams

!         call write_raytrace_camera(c(m),12,outfile,cflag,m,ncams,size(knames), &
!         knames,kdescs,(/real(c(m)%nx),real(c(m)%ny),real(FREQS(1+mod(m-1,nfreq)))/))

          call kwrite_raytrace_camera(c(m),12,outfile,cflag,m,ncams,size(knames), &
         knames,kdescs,(/real(c(m)%nx),real(c(m)%ny),real(FREQS(1+mod(m-1,nfreq)))/), &
         standard,mumin,mumax,nmu,phi0,spin,&
         uout,uin, rcut, nrotype, gridvals, nn, &
         fname, dt, nt, nload, nmdot, mdotmin, mdotmax, &
         ename, mbh, nfreq, fmin, fmax, muval, gmin, gmax,&
         p1, p2, jetalpha, stype, &
         use_geokerr, nvals, iname, extra)

         call del_raytrace_camera(c(m))
       enddo
       call unload_fluid_model(fname)
       deallocate(c)
!       do i=1,nparams
!          deallocate(sparams(i)%gmin)
!       enddo
       deallocate(sparams)
       call delete_inputs()
       return
       end subroutine grtrans_main