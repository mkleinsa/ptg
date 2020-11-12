
#include <iostream>
#include <fstream>
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist, BH)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// [[Rcpp::export]]
void ptg(arma::vec Y,
         arma::vec A,
         arma::mat M,
         arma::mat C,
         int iter,
         int warmup,
         int thin,
         int seed, 
         arma::mat& samples_betam,
         arma::mat& samples_alphaa,
         arma::mat& samples_betaa
         ) {
  
  std::ofstream myfile;
  myfile.open ("ptgdraws.txt");
  
  int n = M.n_rows;
  int p = M.n_cols; // 0 < j < p
  int q = C.n_cols;
  
  double ha = 2.0;
  double la = 1.0;
  double h1 = 2.0;
  double l1 = 1.0;
  double h2 = 2.0;
  double l2 = 1.0;
  double km = 2.0;
  double lm = 1.0;
  double kma = 2.0;
  double lma = 1.0;
  
  arma::vec betam_til(p, arma::fill::zeros);
  arma::vec alphaa_til(p, arma::fill::zeros);
  
  arma::vec betam(p, arma::fill::zeros);
  arma::vec alphaa(p, arma::fill::zeros);
  
  arma::vec betac(q, arma::fill::zeros);
  arma::mat alphac(q, p, arma::fill::zeros);
  
  double lambda2 = 1.0;
  double lambda1 = 1.0;
  double lambda0 = 1.0;
  
  double tausq_b = 1.0;
  double tausq_a = 1.0;
  
  double B1 = 1, B2 = 1, B3 = 1;
  double A1 = 1, A2 = 1, A3 = 1;
  
  double sigmasqe = 1.0;
  double sigmasqa = 1.0;
  double sigmasqg = 1.0;
  double sigmasq1 = 1.0;
  
  //NumericVector sseq = NumericVector::create(1, 2, 3);
  arma::vec sseq(3, arma::fill::zeros);
  sseq(0) = 1.0;
  sseq(1) = 2.0;
  sseq(2) = 3.0;
  
  double betaa = 0.0;
  
  arma::vec prop(3, arma::fill::zeros);
  arma::vec propA(3, arma::fill::zeros);
  int Bk = 0;
  int Ak = 0;
  
  double inf = std::numeric_limits<double>::infinity();
  
  int draw = 0;
  for(int it = 0; it < iter; it++) {
    // sample (beta_m)j
    double u_betam_tillj = 0.0;
    for(int nj = 0; nj < p; nj++) {
      //myfile << "alphaa_til(nj): " << alphaa_til(nj) << "\n";
      if(alphaa_til(nj) == 0.0) {
        u_betam_tillj = lambda1;
      } else {
        arma::vec lambmin(2, arma::fill::zeros);
        lambmin(0) = lambda1;
        lambmin(1) = lambda0/abs(alphaa_til(nj));
        u_betam_tillj = arma::min(lambmin);
        //myfile << "u_betam_tillj: " << u_betam_tillj << "\n";
      }

      double mu_mj = 0.0;
      double sigsq_mj = 0.0;
      for(int ui = 0; ui < n; ui++) {
        mu_mj += as_scalar(M(ui, nj) * (Y(ui) - A(ui) * betaa - M.row(ui) * betam -
          C.row(ui) * betac + M(ui, nj) * betam(nj)));
      }
      mu_mj /= ((sigmasqe / tausq_b) + accu(M.col(nj) % M.col(nj)));
      myfile << "mu_mj: " << mu_mj << "\n";
      sigsq_mj = 1 / ((1 / tausq_b) + (accu(M.col(nj) % M.col(nj)) / sigmasqe));
      myfile << "sigsq_mj: " << sigsq_mj << "\n";
      
      B1 = 1 - 2 * R::pnorm(-1 * (u_betam_tillj / std::sqrt(tausq_b)), 0.0, 1.0, true, false);
      myfile << "B1: " << B1 << "\n";
      
      
      NumericVector compsB2 = {((mu_mj * mu_mj) / (2 * (sigsq_mj))), 
                             log(std::sqrt(sigsq_mj)),
                             -log(std::sqrt(tausq_b))};
      myfile << "B2 part 1: " << compsB2(0) << "\n";
      myfile << "B2 part 2: " << compsB2(1) << "\n";
      myfile << "B2 part 3: " << compsB2(2) << "\n";
      myfile << "B2 cdf: " << log(1 - R::pnorm(u_betam_tillj / std::sqrt(sigsq_mj), mu_mj, 1.0, true, false)) << "\n";
      
      B2 = exp(sum(compsB2) - max(compsB2) + log(1 - R::pnorm(u_betam_tillj / std::sqrt(sigsq_mj), mu_mj, 1.0, true, false)));
      
      NumericVector compsB3 = {((mu_mj * mu_mj) / (2 * (sigsq_mj))), 
                             log(std::sqrt(sigsq_mj)),
                             -log(std::sqrt(tausq_b))};
      
      myfile << "B3 part 1: " << compsB3(0) << "\n";
      myfile << "B3 part 2: " << compsB3(1) << "\n";
      myfile << "B3 part 3: " << compsB3(2) << "\n";
      myfile << "B3 cdf: " << log(R::pnorm((-u_betam_tillj / std::sqrt(sigsq_mj)), mu_mj, 1.0, true, false)) << "\n";
      
      B3 = exp(sum(compsB3) - max(compsB3) + log(R::pnorm((-u_betam_tillj / std::sqrt(sigsq_mj)), mu_mj, 1.0, true, false)));
      
      
      // B2 = exp(((mu_mj * mu_mj) / (2 * (sigsq_mj))) + log(std::sqrt(sigsq_mj)) -
      //   log(std::sqrt(tausq_b))) * (1 - R::pnorm(u_betam_tillj / std::sqrt(sigsq_mj), mu_mj, 1.0, true, false));
      // B3 = exp(((mu_mj * mu_mj) / (2 * (sigsq_mj))) + log(std::sqrt(sigsq_mj)) -
      //   log(std::sqrt(tausq_b))) * (R::pnorm((-u_betam_tillj / std::sqrt(sigsq_mj)), mu_mj, 1.0, true, false)); // is it mu_mj or is it -mu_mj
      myfile << "B2: " << B2 << "\n";
      myfile << "B3: " << B3 << "\n";
      
      if(B1 > 10000) { B1 = 10000; }
      if(B2 > 10000) { B2 = 10000; }
      if(B3 > 10000) { B3 = 10000; }
      
      double sumB = 0;
      sumB = B1 + B2 + B3;
      prop(0) = B1 / sumB;
      prop(1) = B2 / sumB;
      prop(2) = B3 / sumB;
      arma::vec Bk_samp(1, arma::fill::zeros);
      // for(int g = 0; g < 3; g++) {
      //   if(prop(g) < 0.00001) prop(g) = 0;
      // }
      myfile << "prop(0): " << prop(0) << "\n";
      myfile << "prop(1): " << prop(1) << "\n";
      myfile << "prop(2): " << prop(2) << "\n";
      
      Bk_samp = RcppArmadillo::sample(sseq, 1, true, prop);
      myfile << "Bk_samp: " << Bk_samp(0) << "\n";
      Bk = Bk_samp(0);
      
      // sample betam_til from truncated normal distribution
      //myfile << "abs(betam_til(nj)): " << abs(betam_til(nj)) << "\n";
      
      //abs(betam_til(nj)) < u_betam_tillj
      //betam_til(nj) >= u_betam_tillj
      //
      if (Bk == 1) {
        betam_til(nj) = r_truncnorm(0, std::sqrt(tausq_b), -u_betam_tillj, u_betam_tillj);
        //myfile << "betam_til(nj): " << betam_til(nj) << "\n";
      } else if (Bk == 2) {
        betam_til(nj) = r_truncnorm(mu_mj, std::sqrt(sigsq_mj), u_betam_tillj, inf);
        //myfile << "betam_til(nj): " << betam_til(nj) << "\n";
      } else if (Bk == 3) {
        betam_til(nj) = r_truncnorm(mu_mj, std::sqrt(sigsq_mj), -inf, -u_betam_tillj);
        //myfile << "betam_til(nj): " << betam_til(nj) << "\n";
      }
      
      if(abs(betam_til(nj)) < u_betam_tillj) {
        betam(nj) = 0.0;
        //myfile << "betam(nj): " << betam(nj) << "\n";
      } else if(betam_til(nj) >= u_betam_tillj) {
        betam(nj) = betam_til(nj);
        //myfile << "betam(nj): " << betam(nj) << "\n";
      } else if(betam_til(nj) <= -u_betam_tillj){
        betam(nj) = betam_til(nj);
        //myfile << "betam(nj): " << betam(nj) << "\n";
      }
      samples_betam(draw, nj) = betam(nj);
    }
    // sample (alpha_a)j
    double u_alphaa_tillj = 0.0;
    for(int nj = 0; nj < p; nj++) {
      if(betam_til(nj) == 0.0) {
        u_alphaa_tillj = lambda2;
      } else {
        arma::vec lambmin(2, arma::fill::zeros);
        lambmin(0) = lambda2;
        lambmin(1) = lambda0/abs(betam_til(nj));
        u_alphaa_tillj = arma::min(lambmin);
      }

      double mu_aj = 0.0;
      double sigsq_aj = 0.0;
      for(int ui = 0; ui < n; ui++) {
        arma::mat CibyAlphac(1, p, arma::fill::zeros);
        CibyAlphac = C.row(ui) * alphac;
        mu_aj += as_scalar(A(ui) * (M(ui, nj) - CibyAlphac.col(nj)));
      }
      mu_aj /= ((sigmasqg / tausq_a) + accu(A % A));
      sigsq_aj = 1 / ((1 / tausq_a) + (accu(A % A) / sigmasqg));

      A1 = 1 - 2 * R::pnorm(-1 * (u_alphaa_tillj / std::sqrt(tausq_a)), 0.0, 1.0, true, false);
      
      
      NumericVector compsA2 = {((mu_aj * mu_aj) / (2 * (sigsq_aj))), 
                             log(std::sqrt(sigsq_aj)),
                             -log(std::sqrt(tausq_a))};
      
      A2 = exp(sum(compsA2) - max(compsA2) + log(1 - R::pnorm(u_alphaa_tillj / std::sqrt(sigsq_aj), mu_aj, 1.0, true, false)));
      
      // A2 = exp(((mu_aj * mu_aj) / (2 * (sigsq_aj))) + log(std::sqrt(sigsq_aj)) -
      //   log(std::sqrt(tausq_a))) * (1 - R::pnorm(u_alphaa_tillj / std::sqrt(sigsq_aj), mu_aj, 1.0, true, false));
      
      NumericVector compsA3 = {((mu_aj * mu_aj) / (2 * (sigsq_aj))), 
                             log(std::sqrt(sigsq_aj)),
                             -log(std::sqrt(tausq_a))};
      
      A3 = exp(sum(compsA3) - max(compsA3) + log(R::pnorm((-u_alphaa_tillj / std::sqrt(sigsq_aj)), mu_aj, 1.0, true, false)));
      
      
      // A3 = exp(((mu_aj * mu_aj) / (2 * (sigsq_aj))) + log(std::sqrt(sigsq_aj)) -
      //   log(std::sqrt(tausq_a))) * (R::pnorm((-u_alphaa_tillj / std::sqrt(sigsq_aj)), mu_aj, 1.0, true, false));
      
      
      
      
      
      if(A1 > 10000) { A1 = 10000; }
      if(A2 > 10000) { A2 = 10000; }
      if(A3 > 10000) { A3 = 10000; }
      
      double sumA = 0;
      sumA = A1 + A2 + A3;
      propA(0) = A1 / sumA;
      propA(1) = A2 / sumA;
      propA(2) = A3 / sumA;
      arma::vec Ak_samp(1, arma::fill::zeros);
      // for(int g = 0; g < 3; g++) {
      //   if(propA(g) < 0.00001) propA(g) = 0;
      // }
      myfile << "propA(0): " << propA(0) << "\n";
      myfile << "propA(1): " << propA(1) << "\n";
      myfile << "propA(2): " << propA(2) << "\n";
      
      Ak_samp = RcppArmadillo::sample(sseq, 1, true, propA);
      myfile << "Ak_samp: " << Ak_samp(0) << "\n";
      Ak = Ak_samp(0);
      
      
      // sample alphaa_til from truncated normal distribution
      // abs(alphaa_til(nj)) < u_alphaa_tillj
      // alphaa_til(nj) >= u_alphaa_tillj
      if (Ak == 1) {
        alphaa_til(nj) = r_truncnorm(0, std::sqrt(tausq_a), -u_alphaa_tillj, u_alphaa_tillj);
      } else if (Ak == 2) {
        alphaa_til(nj) = r_truncnorm(mu_aj, std::sqrt(sigsq_aj), u_alphaa_tillj, inf);
      } else if (Ak == 3) {
        alphaa_til(nj) = r_truncnorm(mu_aj, std::sqrt(sigsq_aj), -inf, -u_alphaa_tillj);
      }

      
      if (abs(alphaa_til(nj)) < u_alphaa_tillj) {
        alphaa(nj) = 0.0;
      } else if (alphaa_til(nj) >= u_alphaa_tillj) {
        alphaa(nj) = alphaa_til(nj);
      } else if (alphaa_til(nj) <= -u_alphaa_tillj){
        alphaa(nj) = alphaa_til(nj);
      }
      samples_alphaa(draw, nj) = alphaa(nj);
    }
    // sample betaa
    double mu_betaa = 0.0;
    double sigsq_betaa = 0.0;
    for(int ui = 0; ui < n; ui++) {
      mu_betaa += as_scalar(A(ui) * (Y(ui) - M.row(ui) * betam - C.row(ui) * betac));
    }
    mu_betaa /= ((sigmasq1 / sigmasqa) + accu(A % A));
    sigsq_betaa = 1 / ((1 / sigmasqa) + (accu(A % A) / sigmasq1));
    betaa = R::rnorm(mu_betaa, std::sqrt(sigsq_betaa));
    samples_betaa(draw, 0) = betaa;
    // sample sigsqa
    sigmasqa = 1.0 / Rcpp::rgamma(1, (0.5 + ha), 1.0 / (((betaa * betaa) / 2) + la))[0];

    // sample sigsqe
    double l_sigsqe = 0.0;
    for(int ui = 0; ui < n; ui++) {
      double lsigres = as_scalar((Y(ui) - M.row(ui) * betam - A(ui) * betaa - C.row(ui) * betac));
      l_sigsqe += lsigres * lsigres;
    }
    sigmasqe = 1.0 / Rcpp::rgamma(1, (n / 2) + h1, 1.0 / ((l_sigsqe / 2) + l1))[0];
    
    //myfile << "l_sigsqe: " << l_sigsqe << "\n";
    myfile << "sigmasqe: " << sigmasqe << "\n";
    // sample sigsqg
    double l_sigsqg = 0.0;
    for(int ui = 0; ui < n; ui++) {
      arma::rowvec lsigsqg(p, arma::fill::zeros);
      lsigsqg = (M.row(ui) - (A(ui) * alphaa).t() - C.row(ui) * alphac);
      l_sigsqg += accu(lsigsqg % lsigsqg);
    }
    sigmasqg = 1.0 / Rcpp::rgamma(1, ((p * n) / 2) + h2, 1.0 / ((l_sigsqg / 2) + l2))[0];
    myfile << "sigmasqg: " << sigmasqg << "\n";
    // sample tausq_b
    double sum_betamjsq = 0.0;
    for(int jp = 0; jp < p; jp++) {
      sum_betamjsq += betam_til(jp) * betam_til(jp);
    }
    tausq_b = 1.0 / Rcpp::rgamma(1, (p / 2) + km, 1.0 / ((sum_betamjsq / 2) + lm))[0];
    myfile << "tausq_b: " << tausq_b << "\n";
    // sample tausq_a
    double sum_alphaajsq = 0.0;
    for(int jp = 0; jp < p; jp++) {
      sum_alphaajsq += alphaa_til(jp) * alphaa_til(jp);
    }
    tausq_a = 1.0 / Rcpp::rgamma(1, (p / 2) + kma, 1.0 / ((sum_alphaajsq / 2) + lma))[0];
    myfile << "tausq_a: " << tausq_a << "\n";
    // sample beta_cw
    for(int w = 0; w < q; w++) {
      double mu_betac_w = 0.0;
      for(int ni = 0; ni < n; ni++) {
        mu_betac_w += as_scalar(C(ni, w) * (Y(ni) - A(ni) * betaa - M.row(ni) *
          betam - C.row(ni) * betac + C(ni, w) * betac(w)));
      }
      betac(w) = R::rnorm(mu_betac_w / accu(C.col(w) % C.col(w)),
            std::sqrt(sigmasqe / accu(C.col(w) % C.col(w))));
      //myfile << "betac(w): " << betac(w) << "\n";
    }

    // sample (alpha_cw)j
    for(int w = 0; w < q; w++) {
      for(int nj = 0; nj < p; nj++) {
        double mu_alphac_wj = 0.0;
        for(int ni = 0; ni < n; ni++) {
          mu_alphac_wj += as_scalar(C(ni, w) * (M(ni, nj) - A(ni) * alphaa(nj) -
            C.row(ni) * alphac.col(nj) + C(ni, w) * alphac(w, nj)));
        }
        mu_alphac_wj /= accu(C.col(w) % C.col(w));
        alphac(w, nj) = R::rnorm(mu_alphac_wj, 
               std::sqrt(sigmasqg / accu(C.col(w) % C.col(w))));
        //myfile << "alphac(w, nj): " << alphac(w, nj) << "\n";
      }
    }
    draw += 1;
    myfile << "draw: " << draw << "\n";
  }
  myfile.close();
}

/*** R
load('data/a.RData')
load('data/C.RData')
load('data/M.RData')
load('data/y.RData')

iter = 100
warmup = 0
thin = 1
p = ncol(M)
q = ncol(C)

samples_betam = matrix(rep(0, (iter - warmup) * p), nrow = (iter - warmup))
samples_alphaa = matrix(rep(0, (iter - warmup) * p), nrow = (iter - warmup))
samples_betaa = matrix(rep(0, (iter - warmup) * 1), nrow = (iter - warmup))

ptg(Y = y, A = a, M, C, iter, warmup, thin, seed = 7794,
    samples_betam, samples_alphaa, samples_betaa)

mean(samples_betaa[(iter/2):iter,])
apply(samples_betam[(iter/2):iter,], 2, mean)
apply(samples_alphaa[(iter/2):iter,], 2, mean)

# lm(M[,1] ~ a)
# lm(y ~ M[,100])
# lm(y ~ a)
*/
