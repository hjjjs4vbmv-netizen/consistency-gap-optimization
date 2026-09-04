import itertools, math, statistics, unittest
import mpmath

def tcdf(t, df):
    x=df/(df+t*t)
    tail=0.5*float(mpmath.betainc(df/2,0.5,0,x,regularized=True))
    return tail if t<0 else 1-tail

def directional(x):
    x=list(map(float,x)); mean=statistics.mean(x); se=statistics.stdev(x)/math.sqrt(len(x)); t=mean/se
    pneg=tcdf(t,len(x)-1); ppos=1-pneg
    return "NEGATIVE_DIRECTION_SUPPORTED" if mean<0 and pneg<.05 else ("POSITIVE_DIRECTION_SUPPORTED" if mean>0 and ppos<.05 else "DIRECTION_UNRESOLVED")
def signflip_p(x):
    x=list(map(float,x)); obs=statistics.mean(x); vals=[statistics.mean(v*s for v,s in zip(x,signs)) for signs in itertools.product((-1,1),repeat=len(x))]
    return sum(v<=obs+1e-15 for v in vals)/len(vals)
def tost(x):
    x=list(map(float,x)); band=math.log(1.03); se=statistics.stdev(x)/math.sqrt(len(x)); df=len(x)-1; mean=statistics.mean(x)
    return tcdf((mean+band)/se,df)>.95 and tcdf((band-mean)/se,df)>.95
class StatisticsTest(unittest.TestCase):
    def test_direction_is_not_overconstrained(self):
        x=[-.08,-.07,-.06,-.05,-.04,-.03,.01,.02]
        self.assertEqual(directional(x),"NEGATIVE_DIRECTION_SUPPORTED")
        self.assertEqual(sum(v<0 for v in x),6)
    def test_signflip_is_robustness_only_and_exact(self):
        self.assertGreaterEqual(signflip_p([-1,-1,-1,1]),1/16)
    def test_tost_independent(self):
        self.assertTrue(tost([0.001,-0.001,0.002,-0.002,0.0,0.001,-0.001,0.0]))
        self.assertFalse(tost([-.08,-.07,-.06,-.05,-.04,-.03,.01,.02]))
    def test_crossed_algebra(self):
        aa,ab,ba,bb=1.,3.,2.,7.
        self.assertEqual(bb-ba-ab+aa,(bb-ab)-(ba-aa))

if __name__=="__main__": unittest.main()
