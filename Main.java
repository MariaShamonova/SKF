package com.metanit;
import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        double start = 1.5;
        double end = 3.3;

        double kf = ikf(start, end, start);
        double gkf = GaussKF(start, end, start);
        System.out.println(kf);

        double[] skfRes = skf(start, end);
        System.out.println(skfRes[0] + " " + skfRes[1]);
        System.out.println();
        System.out.println(gkf);
        double[] GaussSKFRes = GaussSKF(start, end);
        System.out.println(GaussSKFRes[0] + " " + GaussSKFRes[1]);
    }


    private static double[][] array_LUP(double[][] A, double[][] P, int[] counter) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; ++i) C[i] = A[i].clone();

        for (int i = 0; i < n; ++i) {
            double max = 0;
            int index = i;
            for (int j = i; j < n; ++j) {
                if (Math.abs(A[j][i]) > max) {
                    max = Math.abs(A[j][i]);
                    index = j;
                }
            }
            if (index != i) {
                counter[0] += 1;
                double[] temp = P[i];
                P[i] = P[index];
                P[index] = temp;
                double[] tempV = C[i];
                C[i] = C[index];
                C[index] = tempV;
            }
            for (int j = i + 1; j < n; ++j) {
                C[j][i] = C[j][i] / C[i][i];
            }
            for (int j = i + 1; j < n; ++j) {
                for (int k = i + 1; k < n; ++k) {
                    C[j][k] -= C[j][i] * C[i][k];
                }
            }
        }
        return C;
    }

    private static void array_countLU(double[][] C, double[][] L, double[][] U) {
        int n = C.length;
        for (int i = 0; i < n; ++i) C[i][i] += 1;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (i == j) L[i][j] = 1;
                else L[i][j] = C[i][j];
            }
            for (int j = i; j < n; ++j) {
                if (i == j) U[i][i] = C[i][i] - 1;
                else U[i][j] = C[i][j];
            }
        }
    }

    private static double[] array_countSystem(double[][] L, double[][] U, double[] b) {
        int n = L.length;
        double[] x = new double[n];
        double[] y = new double[n];

        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = 0; j < i; ++j) {
                sum += L[i][j] * y[j];
            }
            y[i] = b[i] - sum;
        }

        for (int i = n - 1; i >= 0; --i) {
            double sum = 0;
            for (int j = i + 1; j < n; ++j) {
                sum += U[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / U[i][i];
        }
        return x;
    }

    private static double[] MV(double[][] A, double[] b) {
        int n = A.length;
        double[] res = new double[n];
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = 0; j < n; ++j) {
                sum += A[i][j] * b[j];
            }
            res[i] = sum;
        }
        return res;
    }

    private static double[] array_sys(double[][] A, double[] vec) {
        int n = A.length;
        double[][] P = new double[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) P[i][i] = 1;
                else P[i][j] = 0;
            }
        }
        int[] counter = {0};
        double[][] C = array_LUP(A, P, counter);
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];
        array_countLU(C, L, U);
        double[] b = MV(P, vec);
        double[] res = array_countSystem(L, U, b);

        return res;
    }

    private static double func(double x) {
        return 2 * Math.cos(2.5 * x) * Math.exp(x / 3) + 4 * Math.sin(3.5 * x) * Math.exp(-3 * x) + x;
    }

    interface Func {
        double count(double a, double b, double c, double alpha);
    }

    private static double ikf(double left, double right, double start) {
        int n = 3;
        double[] x = {left, (left + right) / 2, right};

        Func[] moments = new Func[3];
        moments[0] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) / (1 - alpha) - Math.pow(a - c, 1 - alpha) / (1 - alpha);
        moments[1] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) * (b - alpha * b + c) / ((1 - alpha) * (2 - alpha)) -
                Math.pow(a - c, 1 - alpha) * (a - alpha * a + c) / ((1 - alpha) * (2 - alpha));
        moments[2] = (a, b, c, alpha) -> -Math.pow(b - c, 1 - alpha) * (2 * c * c - 2 * c * b * (alpha - 1) + b * b * (alpha - 2) * (alpha - 1)) / ((alpha - 1) * (alpha - 2) * (alpha - 3)) -
                -Math.pow(a - c, 1 - alpha) * (2 * c * c - 2 * c * a * (alpha - 1) + a * a * (alpha - 2) * (alpha - 1)) / ((alpha - 1) * (alpha - 2) * (alpha - 3));

        double[] mu = {moments[0].count(x[0], x[2], start, 1.0 / 3.0), moments[1].count(x[0], x[2], start, 1.0 / 3.0), moments[2].count(x[0], x[2], start, 1.0 / 3.0)};

        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[i][j] = Math.pow(x[j], i);
            }
        }

        double[] A = array_sys(matrix, mu);
        double kf = 0;
        for (int i = 0; i < n; ++i) kf += A[i] * func(x[i]);

        return kf;
    }

    private static double GaussKF(double left, double right, double start) {
        int n = 3;
        double[] x = {left, (left + right) / 2, right};

        Func[] mom = new Func[6];
        mom[0] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) / (1 - alpha) - Math.pow(a - c, 1 - alpha) / (1 - alpha);
        mom[1] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) * (b - alpha * b + c) / ((1 - alpha) * (2 - alpha)) -
                Math.pow(a - c, 1 - alpha) * (a - alpha * a + c) / ((1 - alpha) * (2 - alpha));
        mom[2] = (a, b, c, alpha) -> -Math.pow(b - c, 1 - alpha) * (2 * c * c - 2 * c * b * (alpha - 1) + b * b * (alpha - 2) * (alpha - 1)) / ((alpha - 1) * (alpha - 2) * (alpha - 3)) -
                -Math.pow(a - c, 1 - alpha) * (2 * c * c - 2 * c * a * (alpha - 1) + a * a * (alpha - 2) * (alpha - 1)) / ((alpha - 1) * (alpha - 2) * (alpha - 3));
        mom[3] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) * (6 * c * c * c - 6 * c * c * b * (alpha - 1) + 3 * c * b * b * (alpha - 2) * (alpha - 1) -
                b * b * b * (alpha - 3) * (alpha - 2) * (alpha - 1)) / ((alpha - 4) * (alpha - 3) * (alpha - 2) * (alpha - 1)) -
                Math.pow(a - c, 1 - alpha) * (6 * c * c * c - 6 * c * c * a * (alpha - 1) + 3 * c * a * a * (alpha - 2) * (alpha - 1) -
                        a * a * a * (alpha - 3) * (alpha - 2) * (alpha - 1)) / ((alpha - 4) * (alpha - 3) * (alpha - 2) * (alpha - 1));
        mom[4] = (a, b, c, alpha) -> -Math.pow(b - c, 1 - alpha) * (24 * Math.pow(c, 4) - 24 * Math.pow(c, 3) * b * (alpha - 1) +
                12 * c * c * b * b * (alpha * alpha - 3 * alpha + 2) - 4 * c * Math.pow(b, 3) * (Math.pow(alpha, 3) - 6 * alpha * alpha +
                11 * alpha - 6) + Math.pow(b, 4) * (Math.pow(alpha, 4) - 10 * Math.pow(alpha, 3) + 35 * alpha * alpha - 50 * alpha + 24)) /
                ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * (alpha - 5)) -
                -Math.pow(a - c, 1 - alpha) * (24 * Math.pow(c, 4) - 24 * Math.pow(c, 3) * a * (alpha - 1) +
                        12 * c * c * a * a * (alpha * alpha - 3 * alpha + 2) - 4 * c * Math.pow(a, 3) * (Math.pow(alpha, 3) - 6 * alpha * alpha +
                        11 * alpha - 6) + Math.pow(a, 4) * (Math.pow(alpha, 4) - 10 * Math.pow(alpha, 3) + 35 * alpha * alpha - 50 * alpha + 24)) /
                        ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * (alpha - 5));
        mom[5] = (a, b, c, alpha) -> Math.pow(b - c, 1 - alpha) * (120 * Math.pow(c, 5) - 120 * Math.pow(c, 4) * b * (alpha - 1) + 60 * c * c * c * b * b * (alpha * alpha - 3 * alpha + 2) -
                20 * c * c * b * b * b * (Math.pow(alpha, 3) - 6 * alpha * alpha + 11 * alpha - 6) + 5 * c * Math.pow(b, 4) * (Math.pow(alpha, 4) - 10 * Math.pow(alpha, 3) + 35 * alpha * alpha -
                50 * alpha + 24) - Math.pow(b, 5) * (Math.pow(alpha, 5) - 15 * Math.pow(alpha, 4) + 85 * Math.pow(alpha, 3) - 225 * alpha * alpha + 274 * alpha - 120)) /
                ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * (alpha - 5) * (alpha - 6)) -
                Math.pow(a - c, 1 - alpha) * (120 * Math.pow(c, 5) - 120 * Math.pow(c, 4) * a * (alpha - 1) + 60 * c * c * c * a * a * (alpha * alpha - 3 * alpha + 2) -
                        20 * c * c * a * a * a * (Math.pow(alpha, 3) - 6 * alpha * alpha + 11 * alpha - 6) + 5 * c * Math.pow(a, 4) * (Math.pow(alpha, 4) - 10 * Math.pow(alpha, 3) + 35 * alpha * alpha -
                        50 * alpha + 24) - Math.pow(a, 5) * (Math.pow(alpha, 5) - 15 * Math.pow(alpha, 4) + 85 * Math.pow(alpha, 3) - 225 * alpha * alpha + 274 * alpha - 120)) /
                        ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * (alpha - 5) * (alpha - 6));

        double[] mu = new double[6];
        for (int i = 0; i < 6; ++i) mu[i] = mom[i].count(x[0], x[2], start, 1.0 / 3.0);

        double[][] matrixMu = new double[n][n];
        double[] m = new double[n];
        for (int s = 0; s < n; ++s) {
            for (int j = 0; j < n; ++j) {
                matrixMu[s][j] = mu[j + s];
            }
            m[s] = -mu[n + s];
        }

        double[] as = array_sys(matrixMu, m);
        double[] xs = Kardano(as[2], as[1], as[0]);

        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[i][j] = Math.pow(xs[j], i);
            }
        }

        double[] A = array_sys(matrix, mu);
        double res = 0;
        for (int i = 0; i < n; ++i) res += A[i] * func(xs[i]);

        return res;
    }

    private static double[] Kardano(double a, double b, double c) {
        double Q = (a * a - 3 * b) / 9;
        double R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
        double S = Q * Q * Q - R * R;
        double[] res = new double[3];
        if (S > 0) {
            double phi = Math.acos(R / Math.sqrt(Q * Q * Q)) / 3.0;
            res[0] = -2 * Math.sqrt(Q) * Math.cos(phi) - a / 3.0;
            res[1] = -2 * Math.sqrt(Q) * Math.cos(phi + 2 * Math.PI / 3.0) - a / 3.0;
            res[2] = -2 * Math.sqrt(Q) * Math.cos(phi - 2 * Math.PI / 3.0) - a / 3.0;
        } else System.out.println("S < 0");
        return res;
    }

    private static double[] skf(double a, double b) {
        ArrayList<Integer> steps = new ArrayList<>();
        ArrayList<Double> integrals = new ArrayList<>();

        double accuracy = 1;
        for (int i = 0; i < 3; ++i) {
            int h = (int) Math.pow(2, i);
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (b - a) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += ikf(a + j * step, a + (j + 1) * step, a);
            }
            integrals.add(sum);
            System.out.println(sum);

            if (h == 2) accuracy = Runge(convDouble(integrals), convInt(steps));
            else if (h > 2) accuracy = Richardson(convDouble(integrals), convInt(steps), a, b);
            if (h != 1) System.out.println("Accuracy: " + accuracy);
            System.out.println();
        }
        int opt = optimalStep(convDouble(integrals), convInt(steps), a, b);

        accuracy = 1;
        for (int h = 8; accuracy > 0.000001; h *= 2) {
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (b - a) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += ikf(a + j * step, a + (j + 1) * step, a);
            }
            integrals.add(sum);
            System.out.println(sum);

            accuracy = Richardson(convDouble(integrals), convInt(steps), a, b);
            System.out.println("Accuracy: " + accuracy);
            System.out.println();
        }

        double[] results = {integrals.get(integrals.size() - 1), optimalSKF(a, b, opt)};
        return results;
    }

    private static double optimalSKF(double start, double end, int opt) {
        System.out.println("Found optimal: " + opt + " steps");
        System.out.println("Optimal step: " + ((end - start) / opt));

        ArrayList<Integer> steps = new ArrayList<>();
        ArrayList<Double> integrals = new ArrayList<>();

        for (int i = 0; i < 3; ++i) {
            int h = opt - 3 + i;
            steps.add(h);
            double step = (end - start) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += ikf(start + j * step, start + (j + 1) * step, start);
            }
            integrals.add(sum);
        }

        double accuracy = 1;
        double eps = 0.000001;
        for (int h = opt; accuracy > eps; ++h) {
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (end - start) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += ikf(start + j * step, start + (j + 1) * step, start);
            }
            integrals.add(sum);
            System.out.println(sum);

            accuracy = Richardson(convDouble(integrals), convInt(steps), start, end);
            System.out.println("Accuracy: " + accuracy);
            System.out.println();
        }

        return integrals.get(integrals.size() - 1);
    }

    private static double[] GaussSKF(double a, double b) {
        ArrayList<Integer> steps = new ArrayList<>();
        ArrayList<Double> integrals = new ArrayList<>();

        for (int i = 0; i < 2; ++i) {
            int h = (int) Math.pow(2, i);
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (b - a) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += GaussKF(a + j * step, a + (j + 1) * step, a);
            }
            integrals.add(sum);
            System.out.println(sum);
            System.out.println();
        }

        double accuracy = 1;
        for (int h = 4; accuracy > 0.000001; h *= 2) {
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (b - a) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += GaussKF(a + j * step, a + (j + 1) * step, a);
            }
            integrals.add(sum);
            System.out.println(sum);

            accuracy = Richardson(convDouble(integrals), convInt(steps), a, b);
            System.out.println("Accuracy: " + accuracy);
            System.out.println();
        }

        int opt = optimalStep(convDouble(integrals), convInt(steps), a, b);
        double[] res = {integrals.get(integrals.size() - 1), optimalGaussSKF(a, b, opt)};
        return res;
    }

    private static double optimalGaussSKF(double start, double end, int opt) {
        System.out.println("Found optimal: " + opt + " steps");
        System.out.println("Optimal step: " + ((end - start) / opt));

        ArrayList<Integer> steps = new ArrayList<>();
        ArrayList<Double> integrals = new ArrayList<>();

        double accuracy = 1;
        double eps = 0.000001;
        for (int h = 1; accuracy > eps; ++h) {
            System.out.println(h + " steps:");
            steps.add(h);
            double step = (end - start) / h;
            double sum = 0;
            for (int j = 0; j < h; ++j) {
                sum += GaussKF(start + j * step, start + (j + 1) * step, start);
            }
            integrals.add(sum);
            System.out.println(sum);

            if (h == 2) accuracy = Runge(convDouble(integrals), convInt(steps)) / 10000;
            else if (h > 2) accuracy = Richardson(convDouble(integrals), convInt(steps), start, end);
            if (h != 1) System.out.println("Accuracy: " + accuracy);
            System.out.println();
        }

        return integrals.get(integrals.size() - 1);
    }

    private static double Richardson(double[] integrals, int[] steps, double start, double end) {
        int m = Eitken(steps, integrals);
        System.out.println(m);
        int n = integrals.length;

        double[][] matrix = new double[n][n];
        double[] vec = new double[n];
        for (int i = 0; i < n; ++i) {
            vec[i] = -integrals[i];
            for (int j = 0; j < n - 1; ++j) {
                matrix[i][j] = Math.pow((end - start) / steps[i], m + j);
            }
            matrix[i][n - 1] = -1;
        }

        double[] res = array_sys(matrix, vec);

        return Math.abs(res[n - 1] - integrals[n - 1]);
    }

    private static int Eitken(int[] steps, double[] integrals) {
        if (integrals.length < 3) return 3;

        int n = integrals.length;
        double L = (double) steps[n - 1] / steps[n - 2];
        double sh1 = integrals[n - 3];
        double sh2 = integrals[n - 2];
        double sh3 = integrals[n - 1];
        double m = -Math.log(Math.abs((sh3 - sh2) / (sh2 - sh1))) / Math.log(L);
        if (m < 1) return 3;
        else return (int) m;
    }

    private static double Runge(double[] integrals, int[] steps) {
        int n = steps.length;
        int m = Eitken(steps, integrals);
        double L = (double) steps[n - 2] / steps[n - 1];
        double sh2 = integrals[n - 1];
        double sh1 = integrals[n - 2];
        return Math.abs((sh2 - sh1) / (1 - Math.pow(L, -m)));
    }

    private static int optimalStep(double[] integrals, int[] steps, double a, double b) {
        int L = 2;
        int m = Eitken(steps, integrals);
        double eps = 0.000001;
        double opt = (b - a) / steps[1] * Math.pow(eps * (1 - Math.pow(L, -m)) / Math.abs(integrals[2] - integrals[1]), 1.0 / m);
        return (int) Math.ceil((b - a) / opt);
    }

    private static int[] convInt(ArrayList<Integer> A) {
        int[] res = new int[A.size()];
        for (int i = 0; i < A.size(); ++i) res[i] = A.get(i);
        return res;
    }

    private static double[] convDouble(ArrayList<Double> A) {
        double[] res = new double[A.size()];
        for (int i = 0; i < A.size(); ++i) res[i] = A.get(i);
        return res;
    }
}