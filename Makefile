CC=gcc

INCLUDES=-Iinclude
LIBS=-lm

SRCS=mat.c nn.c
OBJS=$(SRCS:.c=.o)

TARGET=libtoynn.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDES) -fPIC -c $< -o $@

clean:
	$(RM) *.o $(TARGET)
