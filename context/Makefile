CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -pthread
LDFLAGS = -shared -pthread

LIB_NAME = libcontext_mixer.so
LIB_SOURCES = context_mixer.c
LIB_OBJECTS = $(LIB_SOURCES:.c=.o)
LIB_HEADERS = context_mixer.h

EXAMPLE = example
EXAMPLE_SOURCES = example.c
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:.c=.o)

.PHONY: all clean install test

all: $(LIB_NAME) $(EXAMPLE)

$(LIB_NAME): $(LIB_OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(LIB_OBJECTS): %.o: %.c $(LIB_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(EXAMPLE): $(EXAMPLE_OBJECTS) $(LIB_NAME)
	$(CC) -o $@ $(EXAMPLE_OBJECTS) -L. -lcontext_mixer -Wl,-rpath,'$$ORIGIN'

$(EXAMPLE_OBJECTS): %.o: %.c $(LIB_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(LIB_NAME)
	install -d $(DESTDIR)/usr/local/lib
	install -m 755 $(LIB_NAME) $(DESTDIR)/usr/local/lib/
	install -d $(DESTDIR)/usr/local/include
	install -m 644 $(LIB_HEADERS) $(DESTDIR)/usr/local/include/
	ldconfig

test: $(EXAMPLE)
	@echo "Running context mixer test..."
	./$(EXAMPLE)

clean:
	rm -f $(LIB_OBJECTS) $(EXAMPLE_OBJECTS) $(LIB_NAME) $(EXAMPLE)
