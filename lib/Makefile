PHONY.: all

CXXINCLUDES=-I/usr/include -I. -I/opt/cuda/include -I/usr/local/cuda/include
CXXFLAGS=-O2 -fPIC -std=c++11

NVINCLUDES=-I/usr/include -I.
NVFLAGS=-O2 -Xcompiler -fPIC -std=c++11

LIBS=-lfftw3 -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart

all: libct.so

libct.so: fbp.o iradonT.o misc.o
	g++ -shared -o $@ $^ $(LIBS)

%.o:%.cu
	nvcc -c -o $@ $< $(NVFLAGS) $(NVINCLUDES)

%.o:%.cpp
	g++ -c -o $@ $< $(CXXFLAGS) $(CXXINCLUDES)

clean:
	rm -f *.o *.so

