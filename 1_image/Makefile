CC=nvcc
OBJ=imgproc
FLAGS= -O3	

all: $(OBJ).cu
	$(CC) $(FLAGS) -o $(OBJ) $<
clean:
	rm $(OBJ)
cleanall:
	rm $(OBJ) *.err *.out lleo-*
