// Python binding for the Efficient Compression Tool's Deflate compressor
// (see License.txt and zopfli/COPYING). deflate(data) returns a raw Deflate
// stream made with the same settings as `ect -9`; the GIL is released while
// compressing, so callers can run several streams in threads.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <cstring>

extern "C" {
#include "zopfli/deflate.h"
#include "zopfli/zopfli.h"
}

static PyObject *ect_deflate(PyObject *, PyObject *args)
{
  Py_buffer input;
  unsigned mode = 9;
  if (!PyArg_ParseTuple(args, "y*|I", &input, &mode))
    return nullptr;
  if (input.len == 0) {
    PyBuffer_Release(&input);
    return PyBytes_FromStringAndSize("\x03\x00", 2);
  }
  // ECT reads a few bytes past the end of its input.
  unsigned char *copy = (unsigned char *)malloc((size_t)input.len + 16);
  if (!copy) {
    PyBuffer_Release(&input);
    return PyErr_NoMemory();
  }
  memcpy(copy, input.buf, (size_t)input.len);
  memset(copy + input.len, 0, 16);
  size_t size = (size_t)input.len;
  PyBuffer_Release(&input);

  ZopfliOptions options;
  memset(&options, 0, sizeof(options));
  ZopfliInitOptions(&options, mode, 0, 0);
  unsigned char *out = nullptr;
  size_t out_size = 0;
  unsigned char bit_position = 0;
  Py_BEGIN_ALLOW_THREADS
  ZopfliDeflate(&options, 1, copy, size, &bit_position, &out, &out_size);
  Py_END_ALLOW_THREADS
  free(copy);
  if (!out || !out_size) {
    free(out);
    PyErr_SetString(PyExc_RuntimeError, "ECT produced no output");
    return nullptr;
  }
  PyObject *result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)out_size);
  free(out);
  return result;
}

static PyMethodDef methods[] = {
  {"deflate", ect_deflate, METH_VARARGS,
   "deflate(data, mode=9) -> raw Deflate stream compressed like `ect -<mode>`"},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT, "_ectdeflate", nullptr, -1, methods,
  nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__ectdeflate(void) { return PyModule_Create(&module); }
