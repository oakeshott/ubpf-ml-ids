######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : Makefile
# @created     : 木曜日 12 04, 2025 19:54:11 JST
######################################################################

main:
	$(CC) -Wall -fPIC -c svm.c
	$(CC) -O2 -shared svm.o -o svm.so

clean:
	rm -f *.o
