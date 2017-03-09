FC = gfortran
FCNAME = gnu95
LOCAL_FFLAGS = -ffixed-line-length-132 -fopenmp -O3 -W -Wall -fPIC $(FFLAGS)
LOCAL_LIBS = -lgomp $(LIBS)


all: polsynchemis.so radtrans_integrate.so

libgrtrans.a: interpolate.o interpolate_aux.o math.o odepack.o odepack_aux.o phys_constants.o
	ar cru $@ $^

%.o: %.f90
	$(FC) $(LOCAL_FFLAGS) -o $@ -c $<

%.o: %.f
	$(FC) $(LOCAL_FFLAGS) -o $@ -c $<

%.so: %.f90 libgrtrans.a
	f2py -c $< --fcompiler=$(FCNAME) --f90flags="$(LOCAL_FFLAGS)" -m $(patsubst %.f90,%,$<) $(LOCALLIBS) libgrtrans.a

clean:
	-rm -f *.a *.so *.mod *.o *.pyc
