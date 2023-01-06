!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   cir :    Homoclinic Bifurcation in an Electronic Circuit
!                (the same equations as in demo tor)
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION E
      DOUBLE PRECISION P
      DOUBLE PRECISION R
      DOUBLE PRECISION B
      DOUBLE PRECISION C
      DOUBLE PRECISION DRDE
      DOUBLE PRECISION DBDE
      DOUBLE PRECISION DCDE

       E=PAR(1)
       P=2
       R=-6.9142484373744839e-01-7.3260087859986806e-02*E-2.7462906764622484e-03*E*E
       B=1.7910887544563148e+00-3.0554994266613233e-01*E-2.4878961651300643e-02*E*E
       C=7.5841639153560658e-01+5.5761858956235276e-02*E+5.1480291789433611e-03*E*E
       DRDE=-7.3260087859986806e-02-2.7462906764622484e-03*E*2
       DBDE=-3.0554994266613233e-01-2.4878961651300643e-02*E*2
       DCDE=5.5761858956235276e-02+5.1480291789433611e-03*E*2

       F(1)= U(2)
       F(2)= U(3)
       F(3)= U(4)
       F(4)= R*U(1)-P*U(3)-U(1)+B*U(1)**3-C*U(1)**5

       IF(IJAC.EQ.0)RETURN

       DFDU(1,1)=0.0d0
       DFDU(1,2)=1.0d0
       DFDU(1,3)=0.0d0
       DFDU(1,4)=0.0d0

       DFDU(2,1)=0.0d0
       DFDU(2,2)=0.0d0
       DFDU(2,3)=1.0d0
       DFDU(2,4)=0.0d0

       DFDU(3,1)=0.0d0
       DFDU(3,2)=0.0d0
       DFDU(3,3)=0.0d0
       DFDU(3,4)=1.0d0

       DFDU(4,1)=R-1.0d0+3.0d0*B*U(1)**2-5.0d0*C*U(1)**4
       DFDU(4,2)=0.0d0
       DFDU(4,3)=-P
       DFDU(4,4)=0.0d0

      IF(IJAC.EQ.1)RETURN

!      *Parameter derivatives
       DFDP(1,1)=0.0d0
       DFDP(2,1)=0.0d0
       DFDP(3,1)=0.0d0
       DFDP(4,1)=U(1)*DRDE+U(1)**3*DBDE-U(1)**5*DCDE

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

!----------------------------------------------------------------------
! Problem parameters (only PAR(1-9) are available to the user) :

       U(1)=0.0
       U(2)=0.0
       U(3)=0.0
       U(4)=0.0
       PAR(1)=0.0                ! E

!----------------------------------------------------------------------
! If IEQUIB >0 put initial equilibrium in PAR(11+i), i=1,...,NDIM :

      PAR(12) = 0.0
      PAR(13) = 0.0
      PAR(14) = 0.0
      PAR(15) = 0.0


      END SUBROUTINE STPNT

      SUBROUTINE PVLS(NDIM,U,PAR)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
! Homoclinic bifurcations COMMON block needed here :
      COMMON /BLHOM/ ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV
      INTEGER ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV
      INTEGER I

      DOUBLE PRECISION E
      DOUBLE PRECISION P
      DOUBLE PRECISION R
      DOUBLE PRECISION B
      DOUBLE PRECISION C

      E=PAR(1)
      P=2
      R=-6.9142484373744839e-01-7.3260087859986806e-02*E-2.7462906764622484e-03*E*E
      B=1.7910887544563148e+00-3.0554994266613233e-01*E-2.4878961651300643e-02*E*E
      C=7.5841639153560658e-01+5.5761858956235276e-02*E+5.1480291789433611e-03*E*E

      PAR(2)=R
      PAR(3)=B
      PAR(4)=C

! If IEQUIB =0 put analytic equilibrium in PAR(11+i), i=1..NDIM

      IF(IEQUIB.EQ.0)THEN
        DO I=1,NDIM
          PAR(11+I)= 0.0
        ENDDO
      ENDIF

      END SUBROUTINE PVLS

      !----------------------------------------------------------------------
            SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
      !     ---------- ----

            IMPLICIT NONE
            INTEGER, INTENT(IN) :: NDIM,ICP(*),NBC,IJAC
            DOUBLE PRECISION, INTENT(IN) :: PAR(*),U0(NDIM),U1(NDIM)
            DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
            DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)

             FB(1)=U0(1)
             FB(2)=U1(1)
             FB(3)=U0(3)
             FB(4)=U1(3)

            END SUBROUTINE BCND
      !----------------------------------------------------------------------

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT
