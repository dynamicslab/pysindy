c-----------------------------------------------------------------------
C
C  USER SPECIFIED ROUTINES:
C
C     - boundary conditions
C     - initial conditions
C     - variable properties
C     - local acceleration for fluid (a)
C     - forcing function for passive scalar (q)
C     - general purpose routine for checking errors etc.
C
c-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer e,f,eg
c     e = gllel(eg)

      udiff =0.
      utrans=0.
      return
      end
c-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer e,f,eg
c     e = gllel(eg)


c     Note: this is an acceleration term, NOT a force!
c     Thus, ffx will subsequently be multiplied by rho(x,t).


      ffx = 0.0
      ffy = 0.0
      ffz = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer e,f,eg
c     e = gllel(eg)

      qvol   = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'

      character*80 filename

      parameter(lt=lx1*ly1*lz1*lelv)
      common /scrns/ vort(lt*lelv,3),w1(lt),w2(lt)

c     Common variables for velocity gradients
      common /scrns/ dvxdx(lt*lelv), dvxdy(lt*lelv), dummy1(lt*lelv)
      common /scrns/ dvydx(lt*lelv), dvydy(lt*lelv), dummy2(lt*lelv)
      common /scrns/ dvxdxx(lt*lelv), dvxdyy(lt*lelv), lapvx(lt*lelv)
      common /scrns/ dvydxx(lt*lelv), dvydyy(lt*lelv), lapvy(lt*lelv)

c     Common variables for pressure gradients
      common /scrns/         pm1 (lx1,ly1,lz1,lelv)
      common /scrns/         xm0(lx1,ly1,lz1,lelt)
     $,                      ym0(lx1,ly1,lz1,lelt)
      common /scrns/ dpdx(lt*lelv), dpdy(lt*lelv) 

c     Common variables for vorticity
      common /scrns/ vrtx(lt*lelv), vrty(lt*lelv)
      common /scrns/ vrtxx(lt*lelv), vrtyy(lt*lelv), lapvrt(lt*lelv)

      do i = 1,10
         ifxyo=(i.eq.0)  ! Only write grid for first field
         write (filename, '(a, i5.5)') 'cyl0.f', i
         call load_fld(filename)

         ! do something
         ! note: make sure you save the result into arrays which are
         !       dumped by prepost() e.g. T(nx1,ny1,nz1,nelt,ldimt)

!        Compute and save velocity gradients and laplacian
         ifto=.false.    ! Don't save temperature array
         call gradm1(dvxdx, dvxdy, dummy1, vx)  ! u gradients
         call outpost(dvxdx,dvxdy,dummy1,pr,t,'du_')

         call gradm1(dvydx, dvydy, dummy1, vy)  ! v gradients
         call outpost(dvydx,dvydy,dummy1,pr,t,'dv_')

         call gradm1(dvxdxx, dummy1, dummy2, dvxdx)
         call gradm1(dummy1, dvxdyy, dummy2, dvxdy)
         call add3(lapvx, dvxdxx, dvxdyy, lt*lelv)

         call gradm1(dvydxx, dummy1, dummy2, dvydx)
         call gradm1(dummy1, dvydyy, dummy2, dvydy)
         call add3(lapvy, dvydxx, dvydyy, lt*lelv)
         call outpost(lapvx,lapvy,dummy1,pr,t,'lap')

!        Compute and save pressure gradient
         call mappr(pm1,pr,xm0,ym0)     ! Map pressure field to mesh 1
         call gradm1(dpdx, dpdy, dummy1, pm1)
         call outpost(dpdx,dpdy,dummy1,pr,t,'dp_')

         ifto=.true.    ! Save temperature array (is now vorticity)
         call comp_vort3(vort,w1,w2,vx,vy,vz)
         n=nx1*ny1*nz1*nelv
         call copy(t(1,1,1,1,1),vort,n)  ! Put vorticity in temperature field
         call outpost(vx,vy,dummy1,pr,t,'   ')
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,ieg)
c     NOTE ::: This subroutine MAY NOT be called by every process
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      ux=1.0
      uy=0.0
      uz=0.0
      temp=0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine useric (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      ux=1.0
      uy=0.0
      uz=0.0
      temp=0
      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat
      include 'SIZE'
      include 'TOTAL'

c     call platform_timer(0) ! not too verbose
c     call platform_timer(1) ! mxm, ping-pong, and all_reduce timer

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2
      include 'SIZE'
      include 'TOTAL'

c     param(66) = 4.   ! These give the std nek binary i/o and are 
c     param(67) = 4.   ! good default values

c     Set boundary conditions for gmsh lines
      do iel=1,nelv
      do ifc=1,2*ndim
        id_face = bc(5,ifc,iel,1)
        if (id_face.eq.1) then        ! surface 1 for inlet 
           cbc(ifc,iel,1) = 'v  '
        elseif (id_face.eq.2) then    ! surface 2 for outlet
           cbc(ifc,iel,1) = 'O  '
        elseif (id_face.eq.3) then    ! surface 3 for wall
           cbc(ifc,iel,1) = 'W  '
        elseif (id_face.eq.4) then    ! surface 4 for top
           cbc(ifc,iel,1) = 'SYM  '
        elseif (id_face.eq.5) then    ! surface 5 for bottom
           cbc(ifc,iel,1) = 'SYM  '
        endif
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat3
      include 'SIZE'
      include 'TOTAL'
c
      return
      end
c-----------------------------------------------------------------------

c automatically added by makenek
      subroutine usrdat0() 

      return
      end

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end
