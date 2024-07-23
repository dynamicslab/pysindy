!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   cir :    Homoclinic Bifurcation in cubic swift hohenberg
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
      DOUBLE PRECISION D1
      DOUBLE PRECISION D2
      DOUBLE PRECISION D3
      DOUBLE PRECISION U1
      DOUBLE PRECISION U2
      DOUBLE PRECISION U3
      DOUBLE PRECISION U4
      DOUBLE PRECISION U5

       E=PAR(1)
       D1=0
       D3=0
       U2=0
       U4=0
       D2=(-2.001+0.004*E+0.009*E*E)/(1.002+-0.000*E+-0.005*E*E)
       U1=(-1.000+-0.692+-0.075*E+-0.001*E*E)/(1.002+-0.000*E+-0.005*E*E)
       U3=(1.793+-0.296*E+-0.021*E*E)/(1.002+-0.000*E+-0.005*E*E)
       U5=-(0.760+0.061*E+0.008*E*E)/(1.002+-0.000*E+-0.005*E*E)

       F(1)= U(2)
       F(2)= U(3)
       F(3)= U(4)
       F(4)= U1*U(1)+U2*U(1)**2+U3*U(1)**3+U4*U(1)**4+U5*U(1)**5+D1*U(2)+D2*U(3)+D3*U(3)

       IF(IJAC.EQ.0)RETURN

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
      DOUBLE PRECISION D1
      DOUBLE PRECISION D2
      DOUBLE PRECISION D3
      DOUBLE PRECISION U1
      DOUBLE PRECISION U2
      DOUBLE PRECISION U3
      DOUBLE PRECISION U4
      DOUBLE PRECISION U5

      E=PAR(1)
      D1=0
      D3=0
      U2=0
      U4=0
      D2=(-2.001+0.004*E+0.009*E*E)/(1.002+-0.000*E+-0.005*E*E)
      U1=(-1.000+-0.692+-0.075*E+-0.001*E*E)/(1.002+-0.000*E+-0.005*E*E)
      U3=(1.793+-0.296*E+-0.021*E*E)/(1.002+-0.000*E+-0.005*E*E)
      U5=-(0.760+0.061*E+0.008*E*E)/(1.002+-0.000*E+-0.005*E*E)

      PAR(2)=D1
      PAR(3)=D2
      PAR(4)=D3
      PAR(5)=U1
      PAR(6)=U2
      PAR(7)=U3
      PAR(8)=U4
      PAR(9)=U5


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
