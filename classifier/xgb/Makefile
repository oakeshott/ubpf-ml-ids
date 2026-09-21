######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : Makefile
# @created     : 木曜日 12 04, 2025 19:54:11 JST
######################################################################

main:
	$(CC) -Wall -fPIC -c xgb.c
	$(CC) -shared xgb.o -o xgb.so

clean:
	rm -f *.o
