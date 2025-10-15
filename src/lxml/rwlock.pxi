"""Read-write lock implementation.
"""

cdef extern from *:
    """
#include <pythread.h>

#ifndef LXML_ATOMICS_ENABLED
    #define LXML_ATOMICS_ENABLED 1
#endif

#define __lxml_atomic_int_type int
#define __lxml_nonatomic_int_type int

// For standard C atomics, get the headers first so we have ATOMIC_INT_LOCK_FREE
// defined when we decide to use them.
#if LXML_ATOMICS_ENABLED && (defined(__STDC_VERSION__) && \
                        (__STDC_VERSION__ >= 201112L) && \
                        !defined(__STDC_NO_ATOMICS__))
    #include <stdatomic.h>
#endif

#if LXML_ATOMICS_ENABLED && defined(Py_ATOMIC_H)
    // "Python.h" included "pyatomics.h"

    #define __lxml_atomic_add(value, arg)     _Py_atomic_add_int((value), (arg))
    #define __lxml_atomic_incr_relaxed(value) __lxml_atomic_add((value),  1)
    #define __lxml_atomic_decr_relaxed(value) __lxml_atomic_add((value), -1)

    #ifdef __lxml_DEBUG_ATOMICS
        #warning "Using pyatomics.h atomics"
    #endif

#elif LXML_ATOMICS_ENABLED && (defined(__STDC_VERSION__) && \
                        (__STDC_VERSION__ >= 201112L) && \
                        !defined(__STDC_NO_ATOMICS__) && \
                       ATOMIC_INT_LOCK_FREE == 2)
    // C11 atomics are available and  ATOMIC_INT_LOCK_FREE is definitely on
    #undef __lxml_atomic_int_type
    #define __lxml_atomic_int_type atomic_int

    #define __lxml_atomic_add(value, arg)     atomic_fetch_add_explicit((value), (arg), memory_order_relaxed)
    #define __lxml_atomic_incr_relaxed(value) __lxml_atomic_add((value),  1)
    #define __lxml_atomic_decr_relaxed(value) __lxml_atomic_add((value), -1)

    #if defined(__lxml_DEBUG_ATOMICS) && defined(_MSC_VER)
        #pragma message ("Using standard C atomics")
    #elif defined(__lxml_DEBUG_ATOMICS)
        #warning "Using standard C atomics"
    #endif

#elif LXML_ATOMICS_ENABLED && (__GNUC__ >= 5 || (__GNUC__ == 4 && \
                    (__GNUC_MINOR__ > 1 ||  \
                    (__GNUC_MINOR__ == 1 && __GNUC_PATCHLEVEL__ >= 2))))

    /* gcc >= 4.1.2 */
    #define __lxml_atomic_add(value, arg)     __sync_fetch_and_add((value), (arg))
    #define __lxml_atomic_incr_relaxed(value) __sync_fetch_and_add((value), 1)
    #define __lxml_atomic_decr_relaxed(value) __sync_fetch_and_sub((value), 1)

    #ifdef __lxml_DEBUG_ATOMICS
        #warning "Using GNU atomics"
    #endif

#elif LXML_ATOMICS_ENABLED && defined(_MSC_VER)
    /* msvc */
    #include <intrin.h>
    #undef __lxml_atomic_int_type
    #define __lxml_atomic_int_type long
    #undef __lxml_nonatomic_int_type
    #define __lxml_nonatomic_int_type long

    #pragma intrinsic (_InterlockedExchangeAdd, _InterlockedCompareExchangePointer)

    #define __lxml_atomic_add(value, arg) _InterlockedExchangeAdd((value), (arg))
    #define __lxml_atomic_incr_relaxed(value) __lxml_atomic_add((value),  1)
    #define __lxml_atomic_decr_relaxed(value) __lxml_atomic_add((value), -1)

    #ifdef __lxml_DEBUG_ATOMICS
        #pragma message ("Using MSVC atomics")
    #endif

#elif PY_VERSION_HEX >= 0x030d0000
    #undef LXML_ATOMICS_ENABLED
    #define LXML_ATOMICS_ENABLED 0

    static _lxml_nonatomic_int_type __lxml_atomic_add_cs(PyObject *cs, _lxml_atomic_int_type *value, _lxml_nonatomic_int_type arg) {
        _lxml_nonatomic_int_type old_value;
        Py_BEGIN_CRITICAL_SECTION(cs);
        old_value = *value;
        *value = old_value + arg;
        Py_END_CRITICAL_SECTION();
        return old_value;
    }

    #define __lxml_atomic_add(value, arg)   __lxml_atomic_add_cs(__pyx_v_self, value, arg)
    #define __lxml_atomic_incr_relaxed(value) __lxml_atomic_add((value),  1)
    #define __lxml_atomic_decr_relaxed(value) __lxml_atomic_add((value), -1)

    #ifdef __lxml_DEBUG_ATOMICS
        #warning "Not using atomics, using CPython critical section"
    #endif

#else
    #undef LXML_ATOMICS_ENABLED
    #define LXML_ATOMICS_ENABLED 0

    #define __lxml_atomic_add(value, arg)      ((*(value)) += (arg), (*(value) - (arg)))
    #define __lxml_atomic_incr_relaxed(value)  (*(value))++
    #define __lxml_atomic_decr_relaxed(value)  (*(value))--

    #ifdef __lxml_DEBUG_ATOMICS
        #warning "Not using atomics, using the GIL"
    #endif
#endif
    """
    const bint LXML_ATOMICS_ENABLED
    ctypedef int atomic_int "__lxml_atomic_int_type"
    ctypedef int nonatomic_int "__lxml_nonatomic_int_type"

    nonatomic_int atomic_add  "__lxml_atomic_add"          (atomic_int *value, nonatomic_int arg) noexcept
    nonatomic_int atomic_incr "__lxml_atomic_incr_relaxed" (atomic_int *value) noexcept
    nonatomic_int atomic_decr "__lxml_atomic_decr_relaxed" (atomic_int *value) noexcept


cdef const long max_lock_reader_count = 1 << 30


@cython.final
@cython.internal
cdef class RWLock:
    """Read-write lock.

    Uses a critical section to guard lock operations and a PyMutex for write locking.
    """
    cdef nonatomic_int _nreaders
    cdef cython.pymutex _reader_lock
    cdef cython.pymutex _writer_lock

    cdef void lock_read(self) noexcept:
        self._reader_lock.acquire()
        self._nreaders += 1
        if self._nreaders == 1:
            self._writer_lock.acquire()
        self._reader_lock.release()

    cdef void unlock_read(self) noexcept:
        self._reader_lock.acquire()
        self._nreaders -= 1
        if self._nreaders == 0:
            self._writer_lock.release()
        self._reader_lock.release()

    cdef void lock_write(self) noexcept:
        self._writer_lock.acquire()

    cdef void unlock_write(self) noexcept:
        self._writer_lock.release()

    cdef void lock_write_with(self, RWLock second_lock) noexcept:
        """Acquire two locks for writing at the same time.
        """
        # Avoid deadlocks by deterministically locking an arbitrary lock first.
        if self is second_lock:
            self.lock_write()
        elif <void*>self < <void*>second_lock:
            second_lock.lock_write()
            self.lock_write()
        else:
            self.lock_write()
            second_lock.lock_write()

    cdef void unlock_write_with(self, RWLock second_lock) noexcept:
        """Release two locks for writing after locking them at the same time.
        """
        # Avoid deadlocks by deterministically locking an arbitrary lock first.
        if self is second_lock:
            self.unlock_write()
        elif <void*>self < <void*>second_lock:
            self.unlock_write()
            second_lock.unlock_write()
        else:
            second_lock.unlock_write()
            self.unlock_write()
