/*
 * cb_main.c - standalone runner for the GRT code of AGTF30SysDyn / AGTF30SysSS.
 *
 * Build with -DCB_MODEL=AGTF30SysDyn (or AGTF30SysSS) and -DCB_NY=<width of cb_y>,
 * together with the generated model sources and cb_params_table.h (written by
 * gen_param_table.py from the generated headers: every ExportedGlobal CB_*).
 *
 *   runner -p params.txt [-i flight.csv] -o out.bin [-r 1hz|full] [-T stop_s]
 *   runner -l      list every per-run parameter: "NAME numel v1 [v2 ...]" (build-time values)
 *
 *   params.txt  lines "NAME v1 v2 ..." (# comments); every name must be a CB_*
 *               global of this model and give exactly its number of elements.
 *               Names not given keep the build-time (healthy, cruise) value.
 *   flight.csv  header line, then rows t,alt,mn,dT,pla (t from 0, increasing);
 *               fills CB_In_t/Alt/MN/dT/PLA padded with the last row after t_end
 *               (Dyn only). Stop time = last t unless -T.
 *   out.bin     "CBOUT1\n", then a text line "ncols name1 name2 ...\n", then
 *               rows of float64 (little endian): t, cb_y[0..CB_NY-1].
 *               -r full: every major step; -r 1hz: for each integer second k in
 *               [0, t_stop] the last major step with t <= k (t column = native t).
 *   Exit code 0 ok, 2 usage/input error, 3 model error status, 4 stop requested
 *   before the stop time (e.g. the model's non-convergence stop).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CB_STR2(x) #x
#define CB_STR(x) CB_STR2(x)
#define CB_CAT2(a, b) a##b
#define CB_CAT(a, b) CB_CAT2(a, b)
#define CB_HDR CB_STR(CB_MODEL.h)
#include CB_HDR
#include "cb_params_table.h"

#define M_INIT CB_CAT(CB_MODEL, _initialize)
#define M_STEP CB_CAT(CB_MODEL, _step)
#define M_TERM CB_CAT(CB_MODEL, _terminate)
#define M_RTM  CB_CAT(CB_MODEL, _M)
#define M_Y    CB_CAT(CB_MODEL, _Y)

extern const char *cb_out_names; /* cb_params_table.h: space separated output names */

static int fail(const char *msg, const char *arg)
{
    fprintf(stderr, "cb_runner: %s%s%s\n", msg, arg ? " " : "", arg ? arg : "");
    return 2;
}

static const cb_param_t *find_param(const char *name)
{
    int i;
    for (i = 0; i < CB_NPARAMS; i++)
        if (strcmp(cb_params[i].name, name) == 0) return &cb_params[i];
    return NULL;
}

static int read_params(const char *file)
{
    FILE *f = fopen(file, "r");
    char line[1 << 16];
    if (!f) return fail("cannot open parameter file", file);
    while (fgets(line, sizeof line, f)) {
        char *tok = strtok(line, " \t\r\n");
        const cb_param_t *p;
        int n = 0;
        if (!tok || tok[0] == '#') continue;
        p = find_param(tok);
        if (!p) { fclose(f); return fail("unknown parameter", tok); }
        while ((tok = strtok(NULL, " \t\r\n")) != NULL) {
            char *end;
            double v = strtod(tok, &end);
            if (*end) { fclose(f); return fail("bad number for", p->name); }
            if (n >= p->numel) { fclose(f); return fail("too many values for", p->name); }
            p->addr[n++] = v;
        }
        if (n != p->numel) { fclose(f); return fail("wrong number of values for", p->name); }
    }
    fclose(f);
    return 0;
}

#ifdef CB_HAS_FLIGHT
static int read_flight(const char *file, double *tEnd)
{
    FILE *f = fopen(file, "r");
    char line[4096];
    int n = 0, i;
    double t, a, m, d, p;
    if (!f) return fail("cannot open flight file", file);
    if (!fgets(line, sizeof line, f)) { fclose(f); return fail("empty flight file", file); }
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf", &t, &a, &m, &d, &p) != 5) continue;
        if (n >= CB_NMAX) { fclose(f); return fail("flight longer than CB_NMAX rows:", file); }
        if (n > 0 && !(t > CB_In_t[n - 1])) { fclose(f); return fail("time not increasing in", file); }
        if (n == 0 && t != 0.0) { fclose(f); return fail("flight time must start at 0:", file); }
        CB_In_t[n] = t; CB_In_Alt[n] = a; CB_In_MN[n] = m; CB_In_dT[n] = d; CB_In_PLA[n] = p;
        n++;
    }
    fclose(f);
    if (n < 2) return fail("flight needs at least 2 rows:", file);
    for (i = n; i < CB_NMAX; i++) {     /* pad: t_end+1, t_end+2, ... with the last values */
        CB_In_t[i] = CB_In_t[n - 1] + (double)(i - n + 1);
        CB_In_Alt[i] = CB_In_Alt[n - 1]; CB_In_MN[i] = CB_In_MN[n - 1];
        CB_In_dT[i] = CB_In_dT[n - 1]; CB_In_PLA[i] = CB_In_PLA[n - 1];
    }
    *tEnd = CB_In_t[n - 1];
    return 0;
}
#endif

static void write_row(FILE *o, double t, const double *y)
{
    fwrite(&t, sizeof(double), 1, o);
    fwrite(y, sizeof(double), CB_NY, o);
}

int main(int argc, char **argv)
{
    const char *pfile = NULL, *ifile = NULL, *ofile = NULL, *rate = "1hz";
    double tStop = -1.0, tEndFlight = -1.0, h, t, nextK = 0.0;
    double prev[CB_NY], prevT = 0.0;
    int havePrev = 0, full, i, rc = 0;
    long nRows = 0, nSteps = 0;
    FILE *o;
    clock_t c0 = clock();

    if (argc == 2 && !strcmp(argv[1], "-l")) {
        int k, j;
        for (k = 0; k < CB_NPARAMS; k++) {
            int n = cb_params[k].numel > 16 ? 0 : cb_params[k].numel;   /* long vectors: size only */
            printf("%s %d", cb_params[k].name, cb_params[k].numel);
            for (j = 0; j < n; j++) printf(" %.17g", cb_params[k].addr[j]);
            printf("\n");
        }
        return 0;
    }
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) ifile = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) ofile = argv[++i];
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rate = argv[++i];
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) tStop = atof(argv[++i]);
        else return fail("usage: -p params.txt [-i flight.csv] -o out.bin [-r 1hz|full] [-T stop_s]", NULL);
    }
    if (!ofile) return fail("missing -o", NULL);
    full = !strcmp(rate, "full");
    if (!full && strcmp(rate, "1hz")) return fail("bad -r", rate);
    if (pfile && (rc = read_params(pfile)) != 0) return rc;
#ifdef CB_HAS_FLIGHT
    if (!ifile) return fail("missing -i flight.csv", NULL);
    if ((rc = read_flight(ifile, &tEndFlight)) != 0) return rc;
    if (tStop < 0) tStop = tEndFlight;
#else
    if (ifile) return fail("this model takes no flight file", NULL);
#endif
    o = fopen(ofile, "wb");
    if (!o) return fail("cannot write", ofile);
    fprintf(o, "CBOUT1\n%d t %s\n", CB_NY + 1, cb_out_names);

    M_INIT();
#ifdef CB_FIXED_STEP                 /* discrete model without a time base (SS): t = k * step */
    h = CB_FIXED_STEP;
#else
    h = M_RTM->Timing.stepSize0;
#endif
#ifdef CB_DEFAULT_STOP
    if (tStop < 0) tStop = CB_DEFAULT_STOP;       /* model's own stop time (SS: MWS.In.Max_IterSS) */
#endif
    if (!(tStop >= 0) || tStop > 1e8) { fclose(o); return fail("no finite stop time (use -i or -T)", NULL); }
    for (;;) {
        const char *err = rtmGetErrorStatus(M_RTM);
#ifdef CB_FIXED_STEP
        t = (double)nSteps * h;
#else
        t = rtmGetT(M_RTM);
#endif
        if (err != NULL) { fprintf(stderr, "cb_runner: model error at t=%.9g: %s\n", t, err); rc = 3; break; }
        if (t > tStop + 1e-9 * h) break;
        M_STEP();                        /* outputs at t, then advance to t + h */
        nSteps++;
        if (full) { write_row(o, t, (const double *)M_Y.cb_y); nRows++; }
        else {
            while (havePrev && nextK < t && nextK <= tStop) { write_row(o, prevT, prev); nRows++; nextK += 1.0; }
            memcpy(prev, M_Y.cb_y, sizeof prev); prevT = t; havePrev = 1;
        }
        if (rtmGetStopRequested(M_RTM)) {
            fprintf(stderr, "cb_runner: stop requested by the model at t=%.9g\n", t);
            rc = 4; break;
        }
    }
    if (!full && havePrev) {
        while (nextK <= prevT && nextK <= tStop) { write_row(o, prevT, prev); nRows++; nextK += 1.0; }
    }
    M_TERM();
    fclose(o);
    fprintf(stderr, "cb_runner: model=%s steps=%ld rows=%ld t_last=%.9g stop=%.9g cpu_s=%.3f rc=%d\n",
            CB_STR(CB_MODEL), nSteps, nRows, t, tStop, (double)(clock() - c0) / CLOCKS_PER_SEC, rc);
    return rc;
}
