# Evaluation BERT for Vietnamese

## ViGLUE

### Evaluation Results

<table>
<thead>
<tr>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Model</th>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Params</th>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Seed</th>
<th colspan="2" style="text-align:center;">MNLI</th>
<th style="text-align:center;">QNLI</th>
<th style="text-align:center;">RTE</th>
<th style="text-align:center;">VNRTE</th>
<th style="text-align:center;">WNLI</th>
<th style="text-align:center;">SST2</th>
<th style="text-align:center;">VSFC</th>
<th style="text-align:center;">VSMEC</th>
<th colspan="2" style="text-align:center;">MRPC</th>
<th colspan="2" style="text-align:center;">QQP</th>
<th style="text-align:center;">CoLA</th>
<th style="text-align:center;">VToC</th>
</tr>
<tr>
<th style="text-align:center;">Matched</th>
<th style="text-align:center;">Mismatched</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">F1</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">F1</th>
<th style="text-align:center;">MCC</th>
<th style="text-align:center;">Acc.</th>
</tr>
</thead>
<tbody>
<tr><td colspan="18" style="font-weight: bold; text-align:center;">Pretrained-only</td></tr>
<tr>
<td><strong>QuangDuy/modernbert-tiny-checkpoint-55000ba</strong></td>
<td style="text-align:center">34M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">33.2</td>
<td style="text-align:center">33.2</td>
<td style="text-align:center">51.2</td>
<td style="text-align:center">50.7</td>
<td style="text-align:center">53.27</td>
<td style="text-align:center">57.5</td>
<td style="text-align:center">49.1</td>
<td style="text-align:center">31.78</td>
<td style="text-align:center">12.84</td>
<td style="text-align:center">55.6</td>
<td style="text-align:center">66.8</td>
<td style="text-align:center">28.9</td>
<td style="text-align:center">29.2</td>
<td style="text-align:center">1.1</td>
<td style="text-align:center">5.84</td>
</tr>
<tr>
<td><strong>vinai/phobert-base</strong></td>
<td style="text-align:center">135M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">33.5</td>
<td style="text-align:center">33.5</td>
<td style="text-align:center">49.8</td>
<td style="text-align:center">48.6</td>
<td style="text-align:center">47.97</td>
<td style="text-align:center">37.7</td>
<td style="text-align:center">49.6</td>
<td style="text-align:center">11.27</td>
<td style="text-align:center">17.31</td>
<td style="text-align:center">41.7</td>
<td style="text-align:center">32.1</td>
<td style="text-align:center">71.1</td>
<td style="text-align:center">25.0</td>
<td style="text-align:center">-1.3</td>
<td style="text-align:center">4.15</td>
</tr>
<tr>
<td><strong>Fsoft-AIC/videberta-xsmall</strong></td>
<td style="text-align:center">70M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">32.9</td>
<td style="text-align:center">32.9</td>
<td style="text-align:center">50.6</td>
<td style="text-align:center">50.7</td>
<td style="text-align:center">61.17</td>
<td style="text-align:center">46.6</td>
<td style="text-align:center">49.9</td>
<td style="text-align:center">14.81</td>
<td style="text-align:center">18.61</td>
<td style="text-align:center">43.88</td>
<td style="text-align:center">44.37</td>
<td style="text-align:center">18.8</td>
<td style="text-align:center">29.6</td>
<td style="text-align:center">-3.2</td>
<td style="text-align:center">7.09</td>
</tr>
<tr>
<td><strong>jhu-clsp/mmBERT-base</strong></td>
<td style="text-align:center">140M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">31.4</td>
<td style="text-align:center">30.7</td>
<td style="text-align:center">52.3</td>
<td style="text-align:center">49.9</td>
<td style="text-align:center">50.65</td>
<td style="text-align:center">63.7</td>
<td style="text-align:center">51.2</td>
<td style="text-align:center">18.85</td>
<td style="text-align:center">7.35</td>
<td style="text-align:center">43.9</td>
<td style="text-align:center">47.3</td>
<td style="text-align:center">19.9</td>
<td style="text-align:center">29.9</td>
<td style="text-align:center">2.7</td>
<td style="text-align:center">7.31</td>
</tr>
<tr><td colspan="18" style="font-weight: bold; text-align:center;">Fine-tuned</td></tr>
<tr>
<td><strong>QuangDuy/modernbert-tiny-checkpoint-55000ba</strong></td>
<td style="text-align:center">34M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">66.6</td>
<td style="text-align:center">65.7</td>
<td style="text-align:center">74.7</td>
<td style="text-align:center">50.4</td>
<td style="text-align:center">98.41</td>
<td style="text-align:center">49.3</td>
<td style="text-align:center">78.6</td>
<td style="text-align:center">85.98</td>
<td style="text-align:center">31.89</td>
<td style="text-align:center">63.7</td>
<td style="text-align:center">74.8</td>
<td style="text-align:center">82.4</td>
<td style="text-align:center">57.8</td>
<td style="text-align:center">3.7</td>
<td style="text-align:center">38.25</td>
</tr>
<tr>
<td><strong>vinai/phobert-base</strong></td>
<td style="text-align:center">135M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">80.0</td>
<td style="text-align:center">79.2</td>
<td style="text-align:center">87.2</td>
<td style="text-align:center">61.1</td>
<td style="text-align:center">99.84</td>
<td style="text-align:center">62.3</td>
<td style="text-align:center">90.0</td>
<td style="text-align:center">93.14</td>
<td style="text-align:center">56.99</td>
<td style="text-align:center">80.3</td>
<td style="text-align:center">85.1</td>
<td style="text-align:center">87.9</td>
<td style="text-align:center">66.6</td>
<td style="text-align:center">10.3</td>
<td style="text-align:center">81.16</td>
</tr>
<tr>
<td><strong>Fsoft-AIC/videberta-xsmall</strong></td>
<td style="text-align:center">70M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">67.2</td>
<td style="text-align:center">68.0</td>
<td style="text-align:center">79.4</td>
<td style="text-align:center">56.3</td>
<td style="text-align:center">99.64</td>
<td style="text-align:center">62.3</td>
<td style="text-align:center">79.5</td>
<td style="text-align:center">86.13</td>
<td style="text-align:center">28.86</td>
<td style="text-align:center">75.2</td>
<td style="text-align:center">82.0</td>
<td style="text-align:center">80.6</td>
<td style="text-align:center">59.2</td>
<td style="text-align:center">7.4</td>
<td style="text-align:center">19.98</td>
</tr>
<tr>
<td><strong></strong></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
</tr>
<tr>
<td><strong></strong></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
<td style="text-align:center"></td>
</tr>
</tbody>
</table>

## Perplexity
